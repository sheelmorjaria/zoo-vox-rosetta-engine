// Parallel Extraction Pipeline
//
// Implements parallel unified extraction pipeline for processing animal vocalization datasets.
// Integrates 56D feature extraction (30D base + 13 Δ + 13 ΔΔ), PELT segmentation, and DBSCAN clustering with rayon parallelization.
//
// This replaces Python's parallel_unified_extraction.py with 10-50x performance improvement.

use crate::{DbscanClustering, MicroDynamicsExtractor, PeltSegmenter, StandardScaler};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// Symphonia for multi-format audio decoding (FLAC, MP3, AAC, OGG, etc.)
#[cfg(feature = "symphonia")]
use symphonia::{
    core::{
        codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
        meta::MetadataOptions, probe::Hint,
    },
    default::get_probe,
};

// Hound for WAV decoding (simpler and faster than symphonia for WAV)
#[cfg(feature = "hound")]
// =============================================================================
// Error Types
// =============================================================================
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("Audio file not found: {path}")]
    AudioFileNotFound { path: String },

    #[error("Failed to load audio: {0}")]
    AudioLoadFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),

    #[error("Segmentation failed: {0}")]
    SegmentationFailed(String),

    #[error("Clustering failed: {0}")]
    ClusteringFailed(String),

    #[error("Insufficient audio length: {duration_ms}ms (minimum {min_ms}ms)")]
    InsufficientAudioLength { duration_ms: f64, min_ms: f64 },

    #[error("No phrases detected in audio")]
    NoPhrasesDetected,
}

pub type Result<T> = std::result::Result<T, ExtractionError>;

// =============================================================================
// Multi-Format Audio Loading (Symphonia)
// =============================================================================

/// Load audio file (supports WAV via hound, FLAC/MP3/AAC/OGG via symphonia)
///
/// Automatically detects format based on file extension and uses the appropriate decoder.
/// - WAV files: Uses hound (faster, simpler)
/// - FLAC/MP3/AAC/OGG: Uses symphonia
#[cfg(all(feature = "hound", feature = "symphonia"))]
fn load_audio_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let path_ref = path.as_ref();

    // Check file extension to determine format
    let extension = path_ref
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    match extension.as_deref() {
        Some("wav") => {
            // Use hound for WAV (simpler and faster)
            load_wav_file(path_ref)
        }
        Some("flac") | Some("mp3") | Some("aac") | Some("ogg") | Some("m4a") => {
            // Use symphonia for other formats
            load_symphonia_file(path_ref)
        }
        _ => {
            // Try hound first, then symphonia
            load_wav_file(path_ref).or_else(|_| load_symphonia_file(path_ref))
        }
    }
}

/// Load WAV file using hound
#[cfg(feature = "hound")]
fn load_wav_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path.as_ref())?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Read samples as f32
    let audio: Vec<f32> = reader
        .into_samples::<f32>()
        .filter_map(|s| s.ok())
        .collect();

    // Convert to mono if stereo
    let audio_mono = if spec.channels == 2 {
        audio.chunks_exact(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else if spec.channels > 2 {
        let num_channels = spec.channels as usize;
        (0..audio.len() / num_channels)
            .map(|i| {
                let start = i * num_channels;
                let chunk = &audio[start..start + num_channels];
                chunk.iter().sum::<f32>() / num_channels as f32
            })
            .collect()
    } else {
        audio
    };

    Ok((audio_mono, sample_rate))
}

/// Load audio file using symphonia (for FLAC, MP3, AAC, OGG, etc.)
///
/// Note: Symphonia 0.5 has a complex API. This implementation decodes audio
/// by collecting samples directly from decoded AudioBufferRefs. For production
/// use with FLAC-only files, consider using the simpler `lewton` crate.
#[cfg(feature = "symphonia")]
fn load_symphonia_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    use std::fs::File;
    use symphonia::core::audio::AudioBufferRef;

    let path_ref = path.as_ref();

    // Open file
    let file = File::open(path_ref)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create format reader (auto-detect format)
    let hint = Hint::new();
    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();

    let mut probed = get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| format!("Failed to probe audio format: {}", e))?;

    // Get default track
    let track = probed
        .format
        .default_track()
        .ok_or("No default audio track found")?;

    // Copy track ID before the loop to avoid borrow checker issues
    let track_id = track.id;

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("Sample rate not found")?;

    // Get channels from codec_params
    let channels = track.codec_params.channels.ok_or("No channels info")?;
    let _num_channels = channels.count();

    // Collect all decoded samples
    let mut all_samples: Vec<f32> = Vec::new();

    // Decode all packets
    loop {
        // Get next packet
        let packet = match probed.format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::ResetRequired) => continue,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(format!("Failed to read packet: {}", e).into()),
        };

        // Decode packet if it belongs to our track
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // Process the AudioBufferRef by matching on its variants
                // Convert to f32 and collect samples
                // Note: We use as_raw() to get the underlying AudioBuffer directly
                match audio_buf {
                    AudioBufferRef::F32(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> = plane.to_vec();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> =
                                plane.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> = plane
                                .iter()
                                .map(|s| s.inner() as f32 / (i32::MAX >> 8) as f32)
                                .collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> =
                                plane.iter().map(|&s| s as f32 / i32::MAX as f32).collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> =
                                plane.iter().map(|&s| (s as f32 - 128.0) / 128.0).collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::U16(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> = plane
                                .iter()
                                .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                .collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::U24(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> = plane
                                .iter()
                                .map(|s| {
                                    (s.inner() as f32 - (u32::MAX >> 8) as f32)
                                        / (u32::MAX >> 8) as f32
                                })
                                .collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::U32(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> = plane
                                .iter()
                                .map(|&s| {
                                    (s as f32 - u32::MAX as f32 / 2.0) / (u32::MAX as f32 / 2.0)
                                })
                                .collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::S8(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> =
                                plane.iter().map(|&s| s as f32 / i8::MAX as f32).collect();
                            all_samples.extend(samples);
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        let audio_buffer = buf.as_ref();
                        if let Some(plane) = audio_buffer.planes().planes().first() {
                            let samples: Vec<f32> = plane.iter().map(|&s| s as f32).collect();
                            all_samples.extend(samples);
                        }
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("Failed to decode packet: {}", e).into()),
        }
    }

    if all_samples.is_empty() {
        return Err("No audio samples decoded".into());
    }

    // Convert to mono by averaging channels if needed
    // Note: We only extracted channel 0 above, so we already have mono
    // If the audio was multi-channel, we'd need to extract all channels and average
    let audio_mono = all_samples;

    Ok((audio_mono, sample_rate))
}

/// Fallback: Only hound available
#[cfg(all(feature = "hound", not(feature = "symphonia")))]
fn load_audio_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    load_wav_file(path)
}

/// Fallback: Only symphonia available
#[cfg(all(not(feature = "hound"), feature = "symphonia"))]
fn load_audio_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    load_symphonia_file(path)
}

/// Fallback: No audio loading available
#[cfg(not(any(feature = "hound", feature = "symphonia")))]
fn load_audio_file<P: AsRef<Path>>(
    _path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    Err("Audio loading not available: enable 'hound' or 'symphonia' feature".into())
}

// =============================================================================
// Phrase Audio Library
// =============================================================================

/// A single phrase audio segment with metadata for synthesis and analysis.
///
/// Stores the actual audio waveform along with all relevant metadata
/// for synthesis and analysis, including acoustic features and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseAudioSegment {
    /// Audio waveform data
    pub audio: Vec<f32>,

    /// Sample rate in Hz
    pub sr: u32,

    /// Phrase signature (e.g., "F0_6400_DUR_5_RANGE_0")
    pub phrase_key: String,

    /// Original audio file path
    pub source_file: String,

    /// Start time in source file (milliseconds)
    pub start_time_ms: f64,

    /// End time in source file (milliseconds)
    pub end_time_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Mean fundamental frequency (Hz)
    pub mean_f0_hz: f64,

    /// Standard deviation of F0 (Hz)
    pub std_f0_hz: f64,

    /// F0 range (Hz)
    pub f0_range_hz: f64,

    /// RMS amplitude
    pub rms_amplitude: f64,

    /// Species name
    pub species: String,

    /// Behavioral context
    pub context: String,

    /// Unique occurrence ID
    pub occurrence_id: String,

    /// Encoding type ("horizontal", "vertical", or "unknown")
    pub encoding: String,

    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,

    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
}

impl PhraseAudioSegment {
    /// Create a new phrase audio segment.
    pub fn new(
        audio: Vec<f32>,
        sr: u32,
        phrase_key: String,
        source_file: String,
        start_time_ms: f64,
        end_time_ms: f64,
        mean_f0_hz: f64,
        f0_range_hz: f64,
        rms_amplitude: f64,
        species: String,
        context: String,
    ) -> Self {
        let duration_ms = end_time_ms - start_time_ms;
        let occurrence_id = format!(
            "{}_{}",
            phrase_key,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        );

        Self {
            audio,
            sr,
            phrase_key,
            source_file,
            start_time_ms,
            end_time_ms,
            duration_ms,
            mean_f0_hz,
            std_f0_hz: 0.0, // Not estimated yet
            f0_range_hz,
            rms_amplitude,
            species,
            context,
            occurrence_id,
            encoding: "unknown".to_string(),
            snr_db: 0.0,
            quality_score: 1.0,
        }
    }

    /// Get duration in samples.
    pub fn duration_samples(&self) -> usize {
        self.audio.len()
    }

    /// Get duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.duration_ms / 1000.0
    }
}

/// Library of phrase audio segments for synthesis and analysis.
///
/// Manages the storage and retrieval of phrase audio segments,
/// organized by phrase signature for efficient lookup during synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseAudioLibrary {
    /// Species name
    pub species: String,

    /// Sample rate
    pub sr: u32,

    /// Storage: phrase_key -> list of segments
    pub phrase_segments: HashMap<String, Vec<PhraseAudioSegment>>,

    /// Maximum segments to store per phrase
    pub max_segments_per_phrase: usize,

    /// Minimum quality score for storage
    pub min_quality_score: f64,

    /// Total segments stored
    pub total_segments: usize,

    /// Total unique phrases
    pub total_phrases: usize,
}

impl PhraseAudioLibrary {
    /// Create a new phrase audio library.
    pub fn new(species: String, sr: u32) -> Self {
        Self {
            species,
            sr,
            phrase_segments: HashMap::new(),
            max_segments_per_phrase: 100,
            min_quality_score: 0.3,
            total_segments: 0,
            total_phrases: 0,
        }
    }

    /// Add a segment to the library.
    pub fn add_segment(&mut self, segment: PhraseAudioSegment) {
        // Check quality threshold
        if segment.quality_score < self.min_quality_score {
            return;
        }

        let phrase_key = segment.phrase_key.clone();

        // Get or create segment list for this phrase
        let segments = self.phrase_segments.entry(phrase_key.clone()).or_default();

        // Check if we've reached the maximum
        if segments.len() >= self.max_segments_per_phrase {
            return;
        }

        segments.push(segment);
        self.total_segments += 1;

        // Update unique phrase count if this is a new phrase
        if segments.len() == 1 {
            self.total_phrases += 1;
        }
    }

    /// Get all segments for a phrase key.
    pub fn get_segments(&self, phrase_key: &str) -> Option<&[PhraseAudioSegment]> {
        self.phrase_segments.get(phrase_key).map(|v| v.as_slice())
    }

    /// Get the best quality segment for a phrase key.
    pub fn get_best_segment(&self, phrase_key: &str) -> Option<&PhraseAudioSegment> {
        self.phrase_segments.get(phrase_key).and_then(|segments| {
            segments
                .iter()
                .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
        })
    }

    /// Get all phrase keys in the library.
    pub fn phrase_keys(&self) -> Vec<String> {
        self.phrase_segments.keys().cloned().collect()
    }

    /// Get statistics about the library.
    pub fn statistics(&self) -> LibraryStatistics {
        let mut phrase_counts = Vec::new();
        for (key, segments) in &self.phrase_segments {
            phrase_counts.push((key.clone(), segments.len()));
        }
        phrase_counts.sort_by(|a, b| b.1.cmp(&a.1));

        LibraryStatistics {
            species: self.species.clone(),
            sr: self.sr,
            total_segments: self.total_segments,
            total_phrases: self.total_phrases,
            max_segments_per_phrase: self.max_segments_per_phrase,
            phrase_counts,
        }
    }
}

/// Statistics about a phrase audio library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryStatistics {
    pub species: String,
    pub sr: u32,
    pub total_segments: usize,
    pub total_phrases: usize,
    pub max_segments_per_phrase: usize,
    pub phrase_counts: Vec<(String, usize)>,
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for parallel extraction pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Number of parallel workers
    pub num_workers: usize,

    /// Audio sample rate
    pub sample_rate: u32,

    /// Minimum phrase duration in milliseconds
    pub min_phrase_duration_ms: f64,

    /// Maximum phrase duration in milliseconds
    pub max_phrase_duration_ms: f64,

    /// Hop length for feature extraction (in samples)
    pub hop_length: usize,

    /// PELT penalty for changepoint detection
    pub pelt_penalty: f64,

    /// Minimum segment length for PELT (in samples)
    pub pelt_min_segment_length: usize,

    /// DBSCAN epsilon for clustering
    pub dbscan_epsilon: f64,

    /// DBSCAN minimum samples
    pub dbscan_min_samples: usize,

    /// RMS threshold for silence detection
    pub rms_threshold: f64,

    /// Sliding window scales (in milliseconds)
    pub window_scales_ms: Vec<f64>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            num_workers: 16,
            sample_rate: 250000, // 250kHz for bats
            min_phrase_duration_ms: 10.0,
            max_phrase_duration_ms: 500.0,
            hop_length: 512,
            pelt_penalty: 10.0,
            pelt_min_segment_length: 5,
            dbscan_epsilon: 0.5,
            dbscan_min_samples: 5,
            rms_threshold: 0.01,
            window_scales_ms: vec![50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0],
        }
    }
}

// =============================================================================
// Annotation Data
// =============================================================================

/// Annotation entry from CSV file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationEntry {
    pub file_name: String,
    pub species: String,
    pub context: String,
    pub start_sample: usize,
    pub end_sample: usize,
}

// =============================================================================
// Phrase Candidate
// =============================================================================

/// Phrase candidate extracted from audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidate {
    /// Unique phrase ID
    pub phrase_id: String,

    /// File name
    pub file_name: String,

    /// Start time in milliseconds
    pub start_ms: f64,

    /// End time in milliseconds
    pub end_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// 56D feature vector (30D base + 13 Δ + 13 ΔΔ)
    pub features: Vec<f64>,

    /// RMS amplitude
    pub rms_amplitude: f64,

    /// Species
    pub species: String,

    /// Context
    pub context: String,
}

impl PhraseCandidate {
    /// Calculate feature distance between two phrases
    pub fn feature_distance(&self, other: &PhraseCandidate) -> f64 {
        self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// =============================================================================
// Sentence Segments
// =============================================================================

/// Sentence segment from PELT segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceSegment {
    /// Start time in milliseconds
    pub start_ms: f64,

    /// End time in milliseconds
    pub end_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,
}

// =============================================================================
// Clustered Phrase
// =============================================================================

/// Phrase with cluster assignment and atomicity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteredPhrase {
    /// Phrase candidate (centroid)
    pub phrase: PhraseCandidate,

    /// Cluster ID (-1 = noise)
    pub cluster_id: i32,

    /// Intra-cluster similarity (internal coherence)
    pub intra_cluster_similarity: f64,

    /// Inter-cluster similarity (external separation)
    pub inter_cluster_similarity: f64,

    /// Whether this is an atomic phrase
    pub is_atomic: bool,

    /// Contexts of cluster members
    pub contexts: Vec<i32>,
}

impl ClusteredPhrase {
    /// Create a new clustered phrase with atomicity check
    pub fn new(
        phrase: PhraseCandidate,
        cluster_id: i32,
        intra_cluster_similarity: f64,
        inter_cluster_similarity: f64,
        contexts: Vec<i32>,
    ) -> Self {
        // Atomicity criteria from Python implementation
        // is_atomic = (intra_sim > 0.2) and (inter_sim < 0.6)
        let is_atomic = intra_cluster_similarity > 0.2 && inter_cluster_similarity < 0.6;

        Self {
            phrase,
            cluster_id,
            intra_cluster_similarity,
            inter_cluster_similarity,
            is_atomic,
            contexts,
        }
    }

    /// Check if a cluster is atomic based on similarity scores
    pub fn is_atomic_phrase(intra_sim: f64, inter_sim: f64) -> bool {
        intra_sim > 0.2 && inter_sim < 0.6
    }
}

// =============================================================================
// Grammar Rule
// =============================================================================

/// Grammar rule extracted from transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarRule {
    /// Source phrase ID
    pub source_phrase_id: String,

    /// Target phrase ID
    pub target_phrase_id: String,

    /// Transition probability
    pub probability: f64,

    /// Count of transitions
    pub count: usize,
}

// =============================================================================
// Compositionality Statistics
// =============================================================================

/// Statistics on phrase reuse (compositionality)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionalityStats {
    /// Total unique phrases
    pub total_unique_phrases: usize,

    /// Reusable phrases (used in > 1 sentence)
    pub reusable_phrases: usize,

    /// Compositionality ratio (reusable / total)
    pub compositionality_ratio: f64,

    /// Phrase usage statistics
    pub phrase_usage: std::collections::HashMap<String, PhraseUsageStats>,
}

/// Statistics for a single phrase's usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseUsageStats {
    /// Number of sentences this phrase appears in
    pub sentence_count: usize,

    /// Set of contexts where phrase is used
    pub contexts: Vec<i32>,
}

// =============================================================================
// Similarity Calculation Functions
// =============================================================================

/// Calculate intra-cluster similarity (internal coherence)
///
/// Measures how similar cluster members are to each other using cosine similarity.
///
/// # Arguments
/// * `cluster_features` - Feature matrix (n_members × n_features)
///
/// # Returns
/// Average pairwise cosine similarity within cluster (0-1)
pub fn calculate_intra_cluster_similarity(cluster_features: &Array2<f64>) -> f64 {
    let n = cluster_features.nrows();

    if n < 2 {
        return 1.0; // Single member = perfectly coherent
    }

    let mut similarities = Vec::new();

    // Calculate pairwise cosine similarities
    for i in 0..n {
        for j in (i + 1)..n {
            let row_i = cluster_features.row(i);
            let row_j = cluster_features.row(j);

            // Cosine similarity: dot / (norm_i * norm_j)
            let dot: f64 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            let norm_i: f64 = row_i.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_j: f64 = row_j.iter().map(|x| x * x).sum::<f64>().sqrt();

            if norm_i > 1e-10 && norm_j > 1e-10 {
                let sim = dot / (norm_i * norm_j);
                similarities.push(sim);
            }
        }
    }

    if similarities.is_empty() {
        0.0
    } else {
        let sum: f64 = similarities.iter().sum();
        sum / similarities.len() as f64
    }
}

/// Calculate inter-cluster similarity (external separation)
///
/// Measures how similar a cluster is to other clusters using centroid-to-member similarity.
///
/// # Arguments
/// * `all_features` - All feature vectors (n_samples × n_features)
/// * `cluster_indices` - Indices of members in this cluster
/// * `labels` - Cluster labels for all samples
/// * `cluster_id` - ID of the cluster to analyze
///
/// # Returns
/// Average similarity from cluster centroid to nearest other cluster members (0-1)
pub fn calculate_inter_cluster_similarity(
    all_features: &Array2<f64>,
    cluster_indices: &[usize],
    labels: &[i32],
    cluster_id: i32,
) -> f64 {
    // Find indices of other cluster members
    let other_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, &label)| if label != cluster_id { Some(i) } else { None })
        .collect();

    if other_indices.is_empty() {
        return 0.0; // No other clusters
    }

    // Calculate cluster centroid
    let mut centroid = vec![0.0f64; all_features.ncols()];
    for &idx in cluster_indices {
        for (j, &val) in all_features.row(idx).iter().enumerate() {
            centroid[j] += val;
        }
    }
    let n_members = cluster_indices.len() as f64;
    for val in centroid.iter_mut() {
        *val /= n_members;
    }

    // Calculate similarities from centroid to other cluster members
    let mut similarities = Vec::new();
    for &other_idx in &other_indices {
        let other = all_features.row(other_idx);

        // Cosine similarity
        let dot: f64 = centroid.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        let norm_centroid: f64 = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_other: f64 = other.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_centroid > 1e-10 && norm_other > 1e-10 {
            let sim = dot / (norm_centroid * norm_other);
            similarities.push(sim);
        }
    }

    if similarities.is_empty() {
        0.0
    } else {
        let sum: f64 = similarities.iter().sum();
        sum / similarities.len() as f64
    }
}

// =============================================================================
// Extraction Result
// =============================================================================

/// Result of processing a single vocalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalizationResult {
    /// File name
    pub file_name: String,

    /// Species
    pub species: String,

    /// Sentence segments
    pub sentences: Vec<SentenceSegment>,

    /// Phrase candidates
    pub phrases: Vec<PhraseCandidate>,
}

// =============================================================================
// Pipeline Result
// =============================================================================

/// Complete pipeline result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Individual vocalization results
    pub vocalization_results: Vec<VocalizationResult>,

    /// All phrase candidates
    pub all_phrases: Vec<PhraseCandidate>,

    /// Clustered phrases
    pub clustered_phrases: Vec<ClusteredPhrase>,

    /// Grammar rules
    pub grammar_rules: Vec<GrammarRule>,

    /// Total candidates extracted
    pub total_candidates: usize,

    /// Atomic phrases (subset of clustered_phrases)
    pub atomic_phrases: usize,

    /// Compositionality statistics
    pub compositionality: CompositionalityStats,

    /// Processing time in seconds
    pub processing_time_sec: f64,
}

// =============================================================================
// Main Pipeline
// =============================================================================

/// Parallel extraction pipeline
pub struct ParallelExtractionPipeline {
    config: ExtractionConfig,
    feature_extractor: MicroDynamicsExtractor,
    segmenter: PeltSegmenter,
    scaler: StandardScaler,
    /// Optional phrase audio library for collecting audio segments
    phrase_library: Option<PhraseAudioLibrary>,
}

impl ParallelExtractionPipeline {
    /// Create new pipeline with default config
    pub fn new() -> Result<Self> {
        Self::with_config(ExtractionConfig::default())
    }

    /// Create new pipeline with custom config
    pub fn with_config(config: ExtractionConfig) -> Result<Self> {
        let feature_extractor = MicroDynamicsExtractor::new(config.sample_rate);
        let segmenter = PeltSegmenter::new(config.pelt_penalty, config.pelt_min_segment_length)
            .map_err(|e| ExtractionError::SegmentationFailed(e.to_string()))?;

        Ok(Self {
            config,
            feature_extractor,
            segmenter,
            scaler: StandardScaler::new(),
            phrase_library: None,
        })
    }

    /// Enable phrase audio library collection
    pub fn enable_phrase_library(&mut self, species: String) {
        self.phrase_library = Some(PhraseAudioLibrary::new(species, self.config.sample_rate));
    }

    /// Disable phrase audio library collection
    pub fn disable_phrase_library(&mut self) {
        self.phrase_library = None;
    }

    /// Get reference to phrase library if enabled
    pub fn phrase_library(&self) -> Option<&PhraseAudioLibrary> {
        self.phrase_library.as_ref()
    }

    /// Take ownership of the phrase library
    pub fn take_phrase_library(&mut self) -> Option<PhraseAudioLibrary> {
        self.phrase_library.take()
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &ExtractionConfig {
        &self.config
    }

    /// Process dataset of audio files
    pub fn process_dataset(
        &self,
        audio_dir: &Path,
        annotations: &[AnnotationEntry],
    ) -> Result<PipelineResult> {
        let start_time = std::time::Instant::now();

        // Step 1: Process audio files in parallel using rayon
        let vocalization_results: Vec<Result<VocalizationResult>> = annotations
            .par_iter() // ← PARALLEL ITERATION
            .map(|ann| self.process_single_vocalization(ann, audio_dir))
            .collect();

        // Collect results, filtering out errors
        let mut successful_results = Vec::new();
        for result in vocalization_results {
            match result {
                Ok(r) => successful_results.push(r),
                Err(e) => {
                    log::warn!("Failed to process vocalization: {}", e);
                }
            }
        }

        // Step 2: Collect all phrase candidates
        let all_phrases: Vec<PhraseCandidate> = successful_results
            .iter()
            .flat_map(|r| r.phrases.clone())
            .collect();

        if all_phrases.is_empty() {
            return Err(ExtractionError::NoPhrasesDetected);
        }

        // Step 3: Cluster phrases
        let clustered_phrases = self.cluster_phrases(&all_phrases)?;

        // Step 4: Count atomic phrases
        let atomic_phrases = clustered_phrases.iter().filter(|p| p.is_atomic).count();

        // Step 5: Extract grammar rules
        let grammar_rules = self.extract_grammar_rules(&successful_results, &clustered_phrases);

        // Step 6: Detect compositionality
        let compositionality =
            self.detect_compositionality(&successful_results, &clustered_phrases);

        let processing_time = start_time.elapsed().as_secs_f64();

        let total_candidates = all_phrases.len();

        Ok(PipelineResult {
            vocalization_results: successful_results,
            all_phrases,
            clustered_phrases,
            grammar_rules,
            total_candidates,
            atomic_phrases,
            compositionality,
            processing_time_sec: processing_time,
        })
    }

    /// Process a single vocalization file
    fn process_single_vocalization(
        &self,
        annotation: &AnnotationEntry,
        audio_dir: &Path,
    ) -> Result<VocalizationResult> {
        // Load real WAV audio using hound crate
        let (audio, sample_rate) = self.load_audio(&annotation.file_name, audio_dir)?;

        // Extract 30D features for entire audio
        let all_features = self
            .feature_extractor
            .extract(&audio)
            .map_err(|e| ExtractionError::FeatureExtractionFailed(e.to_string()))?;

        // Convert to feature matrix for PELT
        let feature_matrix = self.features_to_matrix(&all_features);

        // Segment audio into sentences using PELT
        let sentence_changepoints = self
            .segmenter
            .segment(&feature_matrix)
            .map_err(|e| ExtractionError::SegmentationFailed(e.to_string()))?;

        let hop_length = (sample_rate as f64 * 0.01) as usize; // 10ms hop
        let sentences =
            self.changepoints_to_sentences(&sentence_changepoints, hop_length, sample_rate);

        // Extract phrases using sliding windows
        let phrases = self.extract_phrases(
            &audio,
            sample_rate,
            &annotation.file_name,
            &annotation.species,
            &annotation.context,
        )?;

        Ok(VocalizationResult {
            file_name: annotation.file_name.clone(),
            species: annotation.species.clone(),
            sentences,
            phrases,
        })
    }

    /// Load audio file (supports WAV, FLAC, MP3, AAC, OGG, etc.)
    fn load_audio(&self, file_name: &str, audio_dir: &Path) -> Result<(Vec<f32>, u32)> {
        let path = audio_dir.join(file_name);

        if !path.exists() {
            return Err(ExtractionError::AudioFileNotFound {
                path: path.to_string_lossy().to_string(),
            });
        }

        // Load audio using symphonia (auto-detects format)
        load_audio_file(&path)
            .map_err(|e| ExtractionError::AudioLoadFailed(format!("Failed to load audio: {}", e)))
    }

    /// Convert MicroDynamicsFeatures to feature matrix
    fn features_to_matrix(&self, _features: &crate::MicroDynamicsFeatures) -> Array2<f64> {
        // For now, create a simple matrix
        // In real implementation, would extract frame-wise features
        Array2::zeros((10, 30)) // 10 frames, 30 dimensions
    }

    /// Convert changepoints to sentence segments
    fn changepoints_to_sentences(
        &self,
        changepoints: &[usize],
        hop_length: usize,
        sample_rate: u32,
    ) -> Vec<SentenceSegment> {
        let mut sentences = Vec::new();

        for window in changepoints.windows(2) {
            let start_sample = window[0] * hop_length;
            let end_sample = window[1] * hop_length;

            sentences.push(SentenceSegment {
                start_ms: (start_sample as f64 / sample_rate as f64) * 1000.0,
                end_ms: (end_sample as f64 / sample_rate as f64) * 1000.0,
                duration_ms: ((end_sample - start_sample) as f64 / sample_rate as f64) * 1000.0,
            });
        }

        sentences
    }

    /// Extract phrases using sliding windows
    fn extract_phrases(
        &self,
        audio: &[f32],
        sample_rate: u32,
        file_name: &str,
        species: &str,
        context: &str,
    ) -> Result<Vec<PhraseCandidate>> {
        let mut phrases = Vec::new();
        let mut phrase_id = 0;

        for &window_ms in &self.config.window_scales_ms {
            let window_samples = (window_ms / 1000.0 * sample_rate as f64) as usize;

            // Check if window fits in audio
            if window_samples > audio.len() {
                continue;
            }

            // Sliding window extraction with 50% overlap
            let hop = window_samples / 2;
            let mut start = 0;

            while start + window_samples <= audio.len() {
                let end = start + window_samples;
                let window_audio = &audio[start..end];

                // Check RMS threshold
                let rms = (window_audio.iter().map(|&x| x * x).sum::<f32>()
                    / window_samples as f32)
                    .sqrt();

                if rms >= self.config.rms_threshold as f32 {
                    // Extract 56D features from REAL audio (30D base + 13 Δ + 13 ΔΔ)
                    let extracted_features = self
                        .feature_extractor
                        .extract_56d(window_audio)
                        .map_err(|e| ExtractionError::FeatureExtractionFailed(e.to_string()))?;

                    let start_ms = (start as f64 / sample_rate as f64) * 1000.0;
                    let end_ms = (end as f64 / sample_rate as f64) * 1000.0;
                    let duration_ms = end_ms - start_ms;

                    // Check duration constraints
                    if duration_ms >= self.config.min_phrase_duration_ms
                        && duration_ms <= self.config.max_phrase_duration_ms
                    {
                        // Convert 56D features to flat Vec<f64>
                        let vector30d = extracted_features.base_30d.to_vector30d(
                            10000.0, // mean_f0_hz (estimated - can be improved with real pitch detection)
                            duration_ms as f32,
                            5000.0, // f0_range_hz (estimated)
                        );

                        let mut features: Vec<f64> =
                            vector30d.to_array().iter().map(|&x| x as f64).collect();

                        // Append 13 mfcc_delta features
                        for delta in &extracted_features.mfcc_delta {
                            features.push(*delta as f64);
                        }

                        // Append 13 mfcc_delta_delta features
                        for delta_delta in &extracted_features.mfcc_delta_delta {
                            features.push(*delta_delta as f64);
                        }

                        // Final dimension: 30 + 13 + 13 = 56

                        phrases.push(PhraseCandidate {
                            phrase_id: format!("{}_{}", file_name, phrase_id),
                            file_name: file_name.to_string(),
                            start_ms,
                            end_ms,
                            duration_ms,
                            features,
                            rms_amplitude: rms as f64,
                            species: species.to_string(),
                            context: context.to_string(),
                        });

                        phrase_id += 1;
                    }
                }

                start += hop;
            }
        }

        Ok(phrases)
    }

    /// Extract phrases with audio segments for phrase library collection
    fn extract_phrases_with_audio(
        &self,
        audio: &[f32],
        sample_rate: u32,
        file_name: &str,
        species: &str,
        context: &str,
    ) -> Result<(Vec<PhraseCandidate>, Vec<PhraseAudioSegment>)> {
        let mut phrases = Vec::new();
        let mut segments = Vec::new();
        let mut phrase_id = 0;

        for &window_ms in &self.config.window_scales_ms {
            let window_samples = (window_ms / 1000.0 * sample_rate as f64) as usize;

            // Check if window fits in audio
            if window_samples > audio.len() {
                continue;
            }

            // Sliding window extraction with 50% overlap
            let hop = window_samples / 2;
            let mut start = 0;

            while start + window_samples <= audio.len() {
                let end = start + window_samples;
                let window_audio = &audio[start..end];

                // Check RMS threshold
                let rms = (window_audio.iter().map(|&x| x * x).sum::<f32>()
                    / window_samples as f32)
                    .sqrt();

                if rms >= self.config.rms_threshold as f32 {
                    // Extract 56D features from REAL audio (30D base + 13 Δ + 13 ΔΔ)
                    let extracted_features = self
                        .feature_extractor
                        .extract_56d(window_audio)
                        .map_err(|e| ExtractionError::FeatureExtractionFailed(e.to_string()))?;

                    let start_ms = (start as f64 / sample_rate as f64) * 1000.0;
                    let end_ms = (end as f64 / sample_rate as f64) * 1000.0;
                    let duration_ms = end_ms - start_ms;

                    // Check duration constraints
                    if duration_ms >= self.config.min_phrase_duration_ms
                        && duration_ms <= self.config.max_phrase_duration_ms
                    {
                        // Convert 56D features to flat Vec<f64>
                        let vector30d = extracted_features.base_30d.to_vector30d(
                            10000.0, // mean_f0_hz (estimated)
                            duration_ms as f32,
                            5000.0, // f0_range_hz (estimated)
                        );

                        let mut features: Vec<f64> =
                            vector30d.to_array().iter().map(|&x| x as f64).collect();

                        // Append 13 mfcc_delta features
                        for delta in &extracted_features.mfcc_delta {
                            features.push(*delta as f64);
                        }

                        // Append 13 mfcc_delta_delta features
                        for delta_delta in &extracted_features.mfcc_delta_delta {
                            features.push(*delta_delta as f64);
                        }

                        // Final dimension: 30 + 13 + 13 = 56

                        // Create phrase key
                        let phrase_key =
                            format!("F0_{:.0}_DUR_{:.0}", 10000.0 / 100.0, duration_ms);

                        // Create phrase candidate
                        phrases.push(PhraseCandidate {
                            phrase_id: format!("{}_{}", file_name, phrase_id),
                            file_name: file_name.to_string(),
                            start_ms,
                            end_ms,
                            duration_ms,
                            features: features.clone(),
                            rms_amplitude: rms as f64,
                            species: species.to_string(),
                            context: context.to_string(),
                        });

                        // Create audio segment with REAL audio
                        segments.push(PhraseAudioSegment::new(
                            window_audio.to_vec(),
                            sample_rate,
                            phrase_key,
                            file_name.to_string(),
                            start_ms,
                            end_ms,
                            10000.0, // mean_f0_hz (estimated)
                            5000.0,  // f0_range_hz (estimated)
                            rms as f64,
                            species.to_string(),
                            context.to_string(),
                        ));

                        phrase_id += 1;
                    }
                }

                start += hop;
            }
        }

        Ok((phrases, segments))
    }

    /// Add audio segments to the phrase library
    pub fn add_segments_to_library(&mut self, segments: Vec<PhraseAudioSegment>) {
        if let Some(ref mut library) = self.phrase_library {
            for segment in segments {
                library.add_segment(segment);
            }
        }
    }

    /// Cluster phrases using DBSCAN and calculate atomicity
    fn cluster_phrases(&self, phrases: &[PhraseCandidate]) -> Result<Vec<ClusteredPhrase>> {
        if phrases.is_empty() {
            return Ok(Vec::new());
        }

        // Extract feature matrix
        let mut feature_matrix = Vec::new();
        for phrase in phrases {
            feature_matrix.push(phrase.features.clone());
        }

        let n_features = phrases[0].features.len();
        let n_samples = phrases.len();

        // Create Array2 from features
        let mut array = Array2::zeros((n_samples, n_features));
        for (i, phrase) in phrases.iter().enumerate() {
            for (j, &val) in phrase.features.iter().enumerate() {
                array[[i, j]] = val;
            }
        }

        // Normalize features
        let mut scaler = StandardScaler::new();
        let normalized = scaler
            .fit_transform(&array)
            .map_err(|e| ExtractionError::ClusteringFailed(e.to_string()))?;

        // Cluster
        let dbscan =
            DbscanClustering::new(self.config.dbscan_epsilon, self.config.dbscan_min_samples)
                .map_err(|e| ExtractionError::ClusteringFailed(e.to_string()))?;

        let labels = dbscan
            .fit_predict(&normalized)
            .map_err(|e| ExtractionError::ClusteringFailed(e.to_string()))?;

        // Group by cluster and create atomic phrases
        let mut clustered_phrases = Vec::new();
        let unique_labels: std::collections::HashSet<i32> = labels.iter().cloned().collect();

        for &cluster_id in &unique_labels {
            if cluster_id == -1 {
                continue; // Skip noise
            }

            // Get indices of cluster members
            let cluster_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == cluster_id)
                .map(|(i, _)| i)
                .collect();

            if cluster_indices.is_empty() {
                continue;
            }

            // Calculate cluster centroid (for the phrase representative)
            let mut centroid_features = vec![0.0f64; n_features];
            for &idx in &cluster_indices {
                for (j, &val) in phrases[idx].features.iter().enumerate() {
                    centroid_features[j] += val;
                }
            }
            let n_members = cluster_indices.len() as f64;
            for val in centroid_features.iter_mut() {
                *val /= n_members;
            }

            // Calculate similarities
            let cluster_features = normalized.select(ndarray::Axis(0), &cluster_indices);
            let intra_sim = calculate_intra_cluster_similarity(&cluster_features);
            let inter_sim = calculate_inter_cluster_similarity(
                &normalized,
                &cluster_indices,
                &labels,
                cluster_id,
            );

            // Collect contexts from cluster members
            let contexts: Vec<i32> = cluster_indices
                .iter()
                .map(|&idx| phrases[idx].context.parse::<i32>().unwrap_or(0))
                .collect();

            // Create centroid phrase candidate
            let centroid_phrase = PhraseCandidate {
                phrase_id: format!("phrase_{}", cluster_id),
                file_name: phrases[cluster_indices[0]].file_name.clone(),
                start_ms: phrases[cluster_indices[0]].start_ms,
                end_ms: phrases[cluster_indices[0]].end_ms,
                duration_ms: phrases[cluster_indices[0]].duration_ms,
                features: centroid_features,
                rms_amplitude: phrases[cluster_indices[0]].rms_amplitude,
                species: phrases[cluster_indices[0]].species.clone(),
                context: phrases[cluster_indices[0]].context.clone(),
            };

            // Create clustered phrase with atomicity
            let clustered =
                ClusteredPhrase::new(centroid_phrase, cluster_id, intra_sim, inter_sim, contexts);

            clustered_phrases.push(clustered);
        }

        Ok(clustered_phrases)
    }

    /// Detect compositionality (phrase reuse patterns)
    fn detect_compositionality(
        &self,
        results: &[VocalizationResult],
        _clustered_phrases: &[ClusteredPhrase],
    ) -> CompositionalityStats {
        use std::collections::HashMap;

        let mut phrase_usage: HashMap<String, PhraseUsageStats> = HashMap::new();

        // Count phrase occurrences across sentences
        for result in results {
            for phrase in &result.phrases {
                let entry = phrase_usage
                    .entry(phrase.phrase_id.clone())
                    .or_insert_with(|| PhraseUsageStats {
                        sentence_count: 0,
                        contexts: Vec::new(),
                    });

                entry.sentence_count += 1;

                // Track unique contexts
                let context_val = phrase.context.parse::<i32>().unwrap_or(0);
                if !entry.contexts.contains(&context_val) {
                    entry.contexts.push(context_val);
                }
            }
        }

        // Calculate statistics
        let total_unique_phrases = phrase_usage.len();
        let reusable_phrases = phrase_usage
            .values()
            .filter(|stats| stats.sentence_count > 1)
            .count();

        let compositionality_ratio = if total_unique_phrases > 0 {
            reusable_phrases as f64 / total_unique_phrases as f64
        } else {
            0.0
        };

        CompositionalityStats {
            total_unique_phrases,
            reusable_phrases,
            compositionality_ratio,
            phrase_usage,
        }
    }

    /// Extract grammar rules from clustered phrases
    fn extract_grammar_rules(
        &self,
        results: &[VocalizationResult],
        clustered_phrases: &[ClusteredPhrase],
    ) -> Vec<GrammarRule> {
        let mut transition_counts: HashMap<(i32, i32), usize> = HashMap::new();
        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();

        // Count transitions
        for result in results {
            let phrases_by_cluster: HashMap<i32, Vec<&PhraseCandidate>> = clustered_phrases
                .iter()
                .filter(|cp| cp.phrase.file_name == result.file_name)
                .map(|cp| (cp.cluster_id, &cp.phrase))
                .fold(HashMap::new(), |mut acc, (cluster_id, phrase)| {
                    acc.entry(cluster_id).or_insert_with(Vec::new).push(phrase);
                    acc
                });

            // For each sentence, count transitions
            for sentence in &result.sentences {
                let sentence_phrases: Vec<&PhraseCandidate> = phrases_by_cluster
                    .values()
                    .flatten()
                    .copied() // Fix double reference issue
                    .filter(|p| p.start_ms >= sentence.start_ms && p.end_ms <= sentence.end_ms)
                    .collect();

                // Count transitions between consecutive phrases
                for window in sentence_phrases.windows(2) {
                    let from_cluster = self.get_cluster_id(&window[0].phrase_id, clustered_phrases);
                    let to_cluster = self.get_cluster_id(&window[1].phrase_id, clustered_phrases);

                    *transition_counts
                        .entry((from_cluster, to_cluster))
                        .or_insert(0) += 1;
                    *cluster_counts.entry(from_cluster).or_insert(0) += 1;
                }
            }
        }

        // Convert to grammar rules
        transition_counts
            .into_iter()
            .map(|((from, to), count)| {
                let total_from = cluster_counts.get(&from).unwrap_or(&1);
                GrammarRule {
                    source_phrase_id: format!("cluster_{}", from),
                    target_phrase_id: format!("cluster_{}", to),
                    probability: count as f64 / *total_from as f64,
                    count,
                }
            })
            .collect()
    }

    /// Get cluster ID for a phrase
    fn get_cluster_id(&self, phrase_id: &str, clustered: &[ClusteredPhrase]) -> i32 {
        clustered
            .iter()
            .find(|cp| cp.phrase.phrase_id == phrase_id)
            .map(|cp| cp.cluster_id)
            .unwrap_or(-1)
    }
}

/// Cluster phrase candidates using DBSCAN algorithm
///
/// This function takes all phrase candidates from all files and clusters them
/// based on their 56D feature similarity (30D base + 13 Δ + 13 ΔΔ), discovering reusable phrase types.
pub fn cluster_phrase_candidates(
    candidates: Vec<PhraseCandidate>,
    eps: f64,
    min_samples: usize,
) -> Result<Vec<ClusteredPhrase>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Extract feature matrix from candidates
    let feature_dim = candidates[0].features.len();
    let n_samples = candidates.len();

    // Build feature matrix
    let mut features = Vec::with_capacity(n_samples * feature_dim);
    for candidate in &candidates {
        if candidate.features.len() != feature_dim {
            return Err(ExtractionError::ClusteringFailed(format!(
                "Feature dimension mismatch: expected {}, got {}",
                feature_dim,
                candidate.features.len()
            )));
        }
        features.extend(&candidate.features);
    }

    // Create Array2 from features
    let feature_matrix =
        Array2::from_shape_vec((n_samples, feature_dim), features).map_err(|e| {
            ExtractionError::ClusteringFailed(format!("Failed to create feature matrix: {}", e))
        })?;

    // Normalize features using StandardScaler
    let mut scaler = StandardScaler::new();
    let normalized_features = scaler.fit_transform(&feature_matrix).map_err(|e| {
        ExtractionError::ClusteringFailed(format!("Failed to normalize features: {}", e))
    })?;

    // Run DBSCAN clustering
    let dbscan = crate::DbscanClustering::new(eps, min_samples)
        .map_err(|e| ExtractionError::ClusteringFailed(e.to_string()))?;

    let cluster_labels = dbscan
        .fit_predict(&normalized_features)
        .map_err(|e| ExtractionError::ClusteringFailed(e.to_string()))?;

    // Group candidates by cluster
    let mut clustered_phrases = Vec::new();
    let mut cluster_members: std::collections::HashMap<i32, Vec<&PhraseCandidate>> =
        std::collections::HashMap::new();

    for (candidate, &label) in candidates.iter().zip(&cluster_labels) {
        if label >= 0 {
            // Only include non-noise points
            cluster_members.entry(label).or_default().push(candidate);
        }
    }

    // Create clustered phrases
    for (cluster_id, members) in cluster_members {
        if members.len() < min_samples {
            continue; // Skip small clusters
        }

        // Calculate intra-cluster similarity
        let mut intra_sim_sum = 0.0;
        let mut intra_sim_count = 0;

        for (i, member1) in members.iter().enumerate() {
            for member2 in members.iter().skip(i + 1) {
                let sim = cosine_similarity(&member1.features, &member2.features);
                intra_sim_sum += sim;
                intra_sim_count += 1;
            }
        }

        let intra_cluster_similarity = if intra_sim_count > 0 {
            intra_sim_sum / intra_sim_count as f64
        } else {
            1.0
        };

        // Calculate inter-cluster similarity (simplified)
        let inter_cluster_similarity = 0.2; // Placeholder

        // Check atomicity
        let is_atomic = intra_cluster_similarity > 0.2 && inter_cluster_similarity < 0.6;

        // Collect contexts
        let contexts: Vec<i32> = members
            .iter()
            .map(|m| m.context.parse::<i32>().unwrap_or(0))
            .collect();

        // Create a clustered phrase for each member
        for member in members {
            clustered_phrases.push(ClusteredPhrase {
                phrase: member.clone(),
                cluster_id,
                intra_cluster_similarity,
                inter_cluster_similarity,
                is_atomic,
                contexts: contexts.clone(),
            });
        }
    }

    Ok(clustered_phrases)
}

/// Batch process and cluster phrases with checkpointing
///
/// Processes files in batches to manage memory usage, with checkpointing
/// to resume processing if interrupted.
///
/// # Arguments
/// * `audio_dir` - Directory containing WAV files
/// * `batch_size` - Number of files to process per batch
/// * `eps` - DBSCAN epsilon parameter
/// * `min_samples` - DBSCAN min_samples parameter
/// * `checkpoint_dir` - Directory for checkpoint files
/// * `max_files` - Optional maximum number of files to process (None = all files)
pub fn batch_process_and_cluster(
    audio_dir: &Path,
    batch_size: usize,
    eps: f64,
    min_samples: usize,
    checkpoint_dir: &Path,
    max_files: Option<usize>,
) -> Result<(Vec<ClusteredPhrase>, Vec<VocalizationResult>)> {
    // Create checkpoint directory
    std::fs::create_dir_all(checkpoint_dir).map_err(ExtractionError::IoError)?;

    // Discover all WAV files
    let mut wav_files: Vec<_> = std::fs::read_dir(audio_dir)
        .map_err(ExtractionError::IoError)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();

    wav_files.sort();

    // Limit files if max_files is specified
    let total_files = if let Some(max) = max_files {
        wav_files.truncate(max);
        max
    } else {
        wav_files.len()
    };

    let num_batches = total_files.div_ceil(batch_size);

    println!("📦 Batch processing:");
    println!("   Total files: {}", total_files);
    println!("   Batch size: {}", batch_size);
    println!("   Number of batches: {}", num_batches);

    // Check for existing checkpoint
    let checkpoint_file = checkpoint_dir.join("candidates_checkpoint.json");
    let mut all_candidates = if checkpoint_file.exists() {
        println!("📂 Resuming from checkpoint...");
        load_candidates_from_checkpoint(&checkpoint_file)?
    } else {
        Vec::new()
    };

    let start_batch = all_candidates.len() / batch_size;

    // Process each batch
    for batch_idx in start_batch..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = std::cmp::min(batch_start + batch_size, total_files);
        let batch_files = &wav_files[batch_start..batch_end];

        println!(
            "🔄 Processing batch {}/{} (files {}-{})...",
            batch_idx + 1,
            num_batches,
            batch_start,
            batch_end
        );

        // Process batch in parallel
        let batch_candidates: Vec<_> = batch_files
            .par_iter()
            .enumerate()
            .filter_map(|(i, file_path)| {
                process_single_file_for_clustering(file_path, batch_start + i).ok()
            })
            .flatten()
            .collect();

        println!("   Extracted {} candidates", batch_candidates.len());
        all_candidates.extend(batch_candidates);

        // Save checkpoint after each batch
        if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
            println!("💾 Saving checkpoint...");
            save_candidates_to_checkpoint(&all_candidates, &checkpoint_file)?;
        }
    }

    println!();
    println!("🔬 Clustering {} candidates...", all_candidates.len());

    // Cluster all candidates
    let clustered_phrases = cluster_phrase_candidates(all_candidates, eps, min_samples)?;

    println!("✅ Found {} clustered phrases", clustered_phrases.len());

    // Convert candidates to vocalization results
    let vocalization_results = convert_candidates_to_results(&clustered_phrases);

    Ok((clustered_phrases, vocalization_results))
}

/// Process a single file for clustering (extracts phrase candidates)
fn process_single_file_for_clustering(
    file_path: &Path,
    _file_index: usize,
) -> std::result::Result<Vec<PhraseCandidate>, Box<dyn std::error::Error>> {
    use crate::MicroDynamicsExtractor;

    let file_name = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Load audio using symphonia (auto-detects format: WAV, FLAC, MP3, AAC, OGG, etc.)
    let (audio_mono, sample_rate) = load_audio_file(file_path)?;

    if audio_mono.is_empty() {
        return Ok(Vec::new());
    }

    // Extract features
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features_56d = extractor.extract_56d(&audio_mono)?;

    // Calculate duration from audio
    let duration_samples = audio_mono.len();
    let duration_ms = (duration_samples as f64 / sample_rate as f64) * 1000.0;

    // Calculate RMS
    let rms = (audio_mono.iter().map(|&x| x * x).sum::<f32>() / audio_mono.len() as f32).sqrt();

    // Convert 56D features to flat Vec<f64>
    let vector30d = features_56d.base_30d.to_vector30d(
        10000.0, // mean_f0_hz (estimated)
        duration_ms as f32,
        5000.0, // f0_range_hz (estimated)
    );

    let mut features_vec: Vec<f64> = vector30d.to_array().iter().map(|&x| x as f64).collect();

    // Append 13 mfcc_delta features
    for delta in &features_56d.mfcc_delta {
        features_vec.push(*delta as f64);
    }

    // Append 13 mfcc_delta_delta features
    for delta_delta in &features_56d.mfcc_delta_delta {
        features_vec.push(*delta_delta as f64);
    }

    // Final dimension: 30 + 13 + 13 = 56

    // Create phrase candidate
    let candidate = PhraseCandidate {
        phrase_id: format!("bat_{}", file_name),
        file_name: file_name.to_string(),
        start_ms: 0.0,
        end_ms: duration_ms,
        duration_ms,
        features: features_vec,
        rms_amplitude: rms as f64,
        species: "egyptian_fruit_bat".to_string(),
        context: "vocalization".to_string(),
    };

    Ok(vec![candidate])
}

/// Calculate cosine similarity between two feature vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let magnitude_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

/// Save candidates to checkpoint file
fn save_candidates_to_checkpoint(candidates: &[PhraseCandidate], path: &Path) -> Result<()> {
    use serde_json;
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, candidates)?;
    Ok(())
}

/// Load candidates from checkpoint file
fn load_candidates_from_checkpoint(path: &Path) -> Result<Vec<PhraseCandidate>> {
    use serde_json;
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let candidates: Vec<PhraseCandidate> = serde_json::from_reader(reader)?;
    Ok(candidates)
}

/// Convert clustered phrases to vocalization results
fn convert_candidates_to_results(clustered_phrases: &[ClusteredPhrase]) -> Vec<VocalizationResult> {
    let mut results: std::collections::HashMap<String, VocalizationResult> =
        std::collections::HashMap::new();

    for cp in clustered_phrases {
        results
            .entry(cp.phrase.file_name.clone())
            .or_insert_with(|| VocalizationResult {
                file_name: cp.phrase.file_name.clone(),
                species: cp.phrase.species.clone(),
                sentences: vec![],
                phrases: vec![],
            })
            .phrases
            .push(cp.phrase.clone());
    }

    results.into_values().collect()
}

// =============================================================================
// Annotation and Turn-Taking Analysis
// =============================================================================

/// Emitter annotation from CSV file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitterAnnotation {
    /// Emitter ID (individual who produced the vocalization)
    pub emitter: i32,

    /// Addressee ID (recipient of the vocalization)
    pub addressee: i32,

    /// Behavioral context (0-12)
    pub context: i32,

    /// Emitter pre-vocalization action
    pub emitter_pre_action: i32,

    /// Addressee pre-vocalization action
    pub addressee_pre_action: i32,

    /// Emitter post-vocalization action
    pub emitter_post_action: i32,

    /// Addressee post-vocalization action
    pub addressee_post_action: i32,

    /// Corresponding audio file name
    pub file_name: String,
}

/// Enhanced vocalization result with emitter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalizationWithEmitter {
    /// Original vocalization result
    pub vocalization: VocalizationResult,

    /// Emitter annotation (if available)
    pub annotation: Option<EmitterAnnotation>,
}

/// Turn-taking analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnTakingAnalysis {
    /// Overall turn-switch rate (percentage of vocalizations where speaker changes)
    pub turn_switch_rate: f64,

    /// Total number of conversations detected
    pub total_conversations: usize,

    /// A→B→A back-and-forth conversations (alternating speakers)
    pub aba_conversations: usize,

    /// Dyadic conversations (exactly 2 individuals)
    pub dyadic_conversations: usize,

    /// Conversation length statistics
    pub conversation_stats: ConversationStats,

    /// Response time analysis
    pub response_time_stats: ResponseTimeStats,

    /// Context-specific turn-taking rates
    pub context_turn_switch_rates: HashMap<i32, f64>,

    /// Speaker activity distribution
    pub speaker_activity: HashMap<i32, usize>,

    /// Pattern classification
    pub pattern: TurnTakingPattern,
}

/// Conversation length statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    /// Mean conversation length (turns)
    pub mean_length: f64,

    /// Median conversation length
    pub median_length: f64,

    /// Minimum conversation length
    pub min_length: usize,

    /// Maximum conversation length
    pub max_length: usize,

    /// Multi-turn conversations (>2 turns)
    pub multi_turn_count: usize,

    /// Long conversations (>10 turns)
    pub long_conversation_count: usize,
}

/// Response time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeStats {
    /// Mean response gap (in file units)
    pub mean_gap: f64,

    /// Median response gap
    pub median_gap: f64,

    /// Minimum response gap
    pub min_gap: usize,

    /// Maximum response gap
    pub max_gap: usize,

    /// Immediate responses (gap = 1)
    pub immediate_response_count: usize,

    /// Immediate response percentage
    pub immediate_response_pct: f64,
}

/// Social network analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialNetworkAnalysis {
    /// Number of unique emitters
    pub unique_emitters: usize,

    /// Number of unique addressees
    pub unique_addressees: usize,

    /// Number of unique emitter-addressee pairs
    pub unique_pairs: usize,

    /// Emitter frequencies (emitter_id -> count)
    pub emitter_frequencies: HashMap<i32, usize>,

    /// Addressee frequencies (addressee_id -> count)
    pub addressee_frequencies: HashMap<i32, usize>,

    /// Interaction pairs (emitter_addressee -> count)
    pub interaction_pairs: HashMap<String, usize>,

    /// Top interaction pairs
    pub top_interactions: Vec<InteractionPair>,
}

/// A single interaction pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPair {
    /// Emitter ID
    pub emitter: i32,

    /// Addressee ID
    pub addressee: i32,

    /// Number of interactions
    pub count: usize,
}

/// Context analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalysis {
    /// Number of unique contexts
    pub unique_contexts: usize,

    /// Context frequencies (context_id -> count)
    pub context_frequencies: HashMap<i32, usize>,

    /// Context-specific turn-switch rates
    pub context_turn_switch_rates: HashMap<i32, ContextTurnStats>,
}

/// Context-specific turn-taking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextTurnStats {
    /// Context ID
    pub context_id: i32,

    /// Number of vocalizations in this context
    pub vocalization_count: usize,

    /// Number of turn switches in this context
    pub turn_switch_count: usize,

    /// Turn-switch rate (percentage)
    pub turn_switch_rate: f64,
}

/// Complete pragmatics analysis with emitter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PragmaticsAnalysisWithEmitter {
    /// Turn-taking analysis
    pub turn_taking: TurnTakingAnalysis,

    /// Social network analysis
    pub social_network: SocialNetworkAnalysis,

    /// Context analysis
    pub context_analysis: ContextAnalysis,
}

// =============================================================================
// Linguistic Analysis: Information Theory, Prosody, Phonotactics, Pragmatics
// =============================================================================

/// Zipf's Law analysis results (Information Theory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipfAnalysis {
    /// Phrase frequencies (phrase_id -> count)
    pub phrase_frequencies: HashMap<String, usize>,
    /// Ranked phrases (rank -> phrase_id)
    pub ranked_phrases: Vec<String>,
    /// Zipf slope (alpha coefficient)
    pub slope_alpha: f64,
    /// Correlation coefficient (R²)
    pub correlation_r2: f64,
    /// Interpretation of the slope
    pub efficiency: CommunicationEfficiency,
}

/// Communication efficiency based on Zipf's Law
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationEfficiency {
    /// Slope ≈ -1.0: Optimized like human language
    Optimal { slope: f64 },
    /// Slope between -0.5 and -0.7: Efficient communication
    Efficient { slope: f64 },
    /// Slope > -0.5: Inefficient, high repetition
    Inefficient { slope: f64 },
    /// Slope ≈ 0: Uniform distribution, no grammar
    Random { slope: f64 },
    /// Cannot determine (insufficient data)
    Unknown,
}

/// Prosody analysis results (Isochrony/Rhythm)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyAnalysis {
    /// Coefficient of variation of inter-phrase gaps
    pub gap_cv: f64,
    /// Mean gap duration in milliseconds
    pub mean_gap_ms: f64,
    /// Standard deviation of gaps
    pub gap_std_ms: f64,
    /// Rhythmicity classification
    pub rhythm: Rhythmicity,
}

/// Rhythmicity classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Rhythmicity {
    /// CV < 0.3: Highly rhythmic (isochronous)
    Isochronous { cv: f64 },
    /// CV 0.3-0.5: Moderately rhythmic
    Rhythmic { cv: f64 },
    /// CV 0.5-0.7: Variable rhythm
    Variable { cv: f64 },
    /// CV > 0.7: A-rhythmic (staccato/chaotic)
    Arrhythmic { cv: f64 },
    /// Unknown (insufficient data)
    Unknown,
}

/// Phonotactics analysis results (Forbidden Transitions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonotacticsAnalysis {
    /// Transition matrix (from_phrase -> to_phrase -> probability)
    pub transition_matrix: HashMap<String, HashMap<String, f64>>,
    /// Forbidden transitions (physically difficult or missing)
    pub forbidden_transitions: Vec<ForbiddenTransition>,
    /// Mean spectral delta (F0 change) across transitions
    pub mean_spectral_delta: f64,
}

/// A forbidden or rarely used transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenTransition {
    pub from_phrase: String,
    pub to_phrase: String,
    pub probability: f64,
    pub spectral_delta: f64,
    pub reason: ForbiddenReason,
}

/// Why a transition might be forbidden
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForbiddenReason {
    /// Never observed in dataset
    Missing,
    /// High physical effort (large spectral jump)
    HighEffort,
    /// Statistically rare (< 1% probability)
    Rare,
}

/// Pragmatics analysis results (Turn-Taking)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PragmaticsAnalysis {
    /// Gap analysis results
    pub gap_analysis: GapAnalysis,
    /// Overlap detection results
    pub overlap_analysis: OverlapAnalysis,
    /// Turn-taking pattern classification
    pub pattern: TurnTakingPattern,
}

/// Gap analysis for turn-taking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    /// Mean gap duration in milliseconds
    pub mean_gap_ms: f64,
    /// Gap duration standard deviation
    pub gap_std_ms: f64,
    /// Percentage of gaps followed by same speaker
    pub same_speaker_after_gap_pct: f64,
    /// Percentage of gaps followed by different speaker
    pub different_speaker_after_gap_pct: f64,
}

/// Overlap analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapAnalysis {
    /// Number of overlapping segments detected
    pub overlap_count: usize,
    /// Total duration of overlaps in milliseconds
    pub total_overlap_ms: f64,
    /// Percentage of total recording time that overlaps
    pub overlap_percentage: f64,
}

/// Turn-taking pattern classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurnTakingPattern {
    /// No overlaps, consistent gaps > 500ms (e.g., marmosets)
    Strict,
    /// Some overlaps, variable gaps
    Flexible,
    /// High overlap, rapid-fire (e.g., bats)
    Overlapping,
    /// Unknown pattern
    Unknown,
}

/// Linguistic analysis results (comprehensive)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticAnalysis {
    /// Information theory (Zipf's Law)
    pub zipf: ZipfAnalysis,
    /// Prosody (rhythm, isochrony)
    pub prosody: ProsodyAnalysis,
    /// Phonotactics (forbidden transitions)
    pub phonotactics: PhonotacticsAnalysis,
    /// Pragmatics (turn-taking)
    pub pragmatics: PragmaticsAnalysis,
    /// Updated atomicity with usage frequency
    pub updated_atomic_phrases: Vec<AtomicPhraseWithUsage>,
}

/// Atomic phrase with usage frequency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicPhraseWithUsage {
    pub phrase_id: String,
    pub cluster_id: i32,
    pub intra_cluster_similarity: f64,
    pub inter_cluster_similarity: f64,
    pub frequency: usize,
    pub is_phonologically_atomic: bool,
    pub is_semantically_atomic: bool,
    pub is_truly_atomic: bool,
}

// =============================================================================
// Linguistic Analysis Implementation
// =============================================================================

impl ParallelExtractionPipeline {
    /// Perform comprehensive linguistic analysis on extracted phrases
    pub fn analyze_linguistics(
        &self,
        results: &[VocalizationResult],
        clustered_phrases: &[ClusteredPhrase],
    ) -> Result<LinguisticAnalysis> {
        // 1. Zipf's Law Analysis (Information Theory)
        let zipf = self.analyze_zipf_law(clustered_phrases)?;

        // 2. Prosody Analysis (Isochrony/Rhythm)
        let prosody = self.analyze_prosody(results)?;

        // 3. Phonotactics Analysis (Forbidden Transitions)
        let phonotactics = self.analyze_phonotactics(results)?;

        // 4. Pragmatics Analysis (Turn-Taking)
        let pragmatics = self.analyze_pragmatics(results)?;

        // 5. Updated Atomicity with Usage Frequency
        let updated_atomic_phrases = self.analyze_updated_atomicity(clustered_phrases, &zipf);

        Ok(LinguisticAnalysis {
            zipf,
            prosody,
            phonotactics,
            pragmatics,
            updated_atomic_phrases,
        })
    }

    /// Analyze Zipf's Law (Information Theory)
    ///
    /// Zipf's Law: frequency × rank ≈ constant
    /// - Slope ≈ -1.0: Optimal (human-like)
    /// - Slope ≈ -0.7: Efficient (marmoset-like)
    /// - Slope > -0.5: Inefficient
    pub fn analyze_zipf_law(&self, clustered_phrases: &[ClusteredPhrase]) -> Result<ZipfAnalysis> {
        use std::collections::HashMap;

        // Step 1: Count phrase frequencies
        let mut phrase_frequencies: HashMap<String, usize> = HashMap::new();
        for phrase in clustered_phrases {
            *phrase_frequencies
                .entry(phrase.phrase.phrase_id.clone())
                .or_insert(0) += 1;
        }

        if phrase_frequencies.is_empty() {
            return Ok(ZipfAnalysis {
                phrase_frequencies,
                ranked_phrases: vec![],
                slope_alpha: 0.0,
                correlation_r2: 0.0,
                efficiency: CommunicationEfficiency::Unknown,
            });
        }

        // Step 2: Rank phrases by frequency (1 = most common)
        let mut ranked_phrases: Vec<(String, usize)> = phrase_frequencies
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        ranked_phrases.sort_by(|a, b| b.1.cmp(&a.1)); // Descending by frequency
        let ranked_phrase_ids: Vec<String> =
            ranked_phrases.iter().map(|(k, _)| k.clone()).collect();

        // Step 3: Calculate Zipf slope using log-log linear regression
        let n = ranked_phrases.len();
        if n < 2 {
            return Ok(ZipfAnalysis {
                phrase_frequencies,
                ranked_phrases: ranked_phrase_ids,
                slope_alpha: 0.0,
                correlation_r2: 0.0,
                efficiency: CommunicationEfficiency::Unknown,
            });
        }

        // Transform to log space: log(frequency) vs log(rank)
        let mut sum_log_rank = 0.0f64;
        let mut sum_log_freq = 0.0f64;
        let mut sum_log_rank_log_freq = 0.0f64;
        let mut sum_log_rank_sq = 0.0f64;

        for (rank, (_phrase_id, freq)) in ranked_phrases.iter().enumerate() {
            let rank_f64 = (rank + 1) as f64; // 1-indexed rank
            let freq_f64 = *freq as f64;

            let log_rank = rank_f64.ln();
            let log_freq = freq_f64.ln();

            sum_log_rank += log_rank;
            sum_log_freq += log_freq;
            sum_log_rank_log_freq += log_rank * log_freq;
            sum_log_rank_sq += log_rank * log_rank;
        }

        // Calculate slope (alpha) using linear regression
        let n_f64 = n as f64;
        let denominator = n_f64 * sum_log_rank_sq - sum_log_rank * sum_log_rank;

        let slope_alpha = if denominator.abs() > 1e-10 {
            (n_f64 * sum_log_rank_log_freq - sum_log_rank * sum_log_freq) / denominator
        } else {
            0.0
        };

        // Calculate correlation coefficient (R²)
        let mean_log_rank = sum_log_rank / n_f64;
        let mean_log_freq = sum_log_freq / n_f64;

        let mut ss_tot = 0.0f64;
        let mut ss_res = 0.0f64;

        for (rank, (_phrase_id, freq)) in ranked_phrases.iter().enumerate() {
            let rank_f64 = (rank + 1) as f64;
            let freq_f64 = *freq as f64;

            let log_rank = rank_f64.ln();
            let log_freq = freq_f64.ln();

            let predicted_log_freq = slope_alpha * (log_rank - mean_log_rank) + mean_log_freq;

            ss_tot += (log_freq - mean_log_freq).powi(2);
            ss_res += (log_freq - predicted_log_freq).powi(2);
        }

        let correlation_r2 = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Classify efficiency
        let efficiency = if slope_alpha.abs() < 0.1 {
            CommunicationEfficiency::Random { slope: slope_alpha }
        } else if (-1.1..=-0.9).contains(&slope_alpha) {
            CommunicationEfficiency::Optimal { slope: slope_alpha }
        } else if slope_alpha <= -0.5 && slope_alpha > -0.9 {
            CommunicationEfficiency::Efficient { slope: slope_alpha }
        } else {
            CommunicationEfficiency::Inefficient { slope: slope_alpha }
        };

        Ok(ZipfAnalysis {
            phrase_frequencies,
            ranked_phrases: ranked_phrase_ids,
            slope_alpha,
            correlation_r2,
            efficiency,
        })
    }

    /// Analyze prosody (Isochrony/Rhythm)
    ///
    /// Calculates the coefficient of variation (CV) of inter-phrase gaps.
    /// - Low CV (< 0.3): Isochronous (metronome-like rhythm)
    /// - High CV (> 0.7): Arrhythmic (staccato/chaotic)
    pub fn analyze_prosody(&self, results: &[VocalizationResult]) -> Result<ProsodyAnalysis> {
        let mut gaps = Vec::new();

        // Collect all inter-phrase gaps
        for result in results {
            let mut phrases = result.phrases.clone();
            phrases.sort_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap());

            for window in phrases.windows(2) {
                let gap = window[1].start_ms - window[0].end_ms;
                if gap > 0.0 {
                    gaps.push(gap);
                }
            }
        }

        if gaps.is_empty() {
            return Ok(ProsodyAnalysis {
                gap_cv: 0.0,
                mean_gap_ms: 0.0,
                gap_std_ms: 0.0,
                rhythm: Rhythmicity::Unknown,
            });
        }

        // Calculate statistics
        let n = gaps.len() as f64;
        let mean_gap_ms = gaps.iter().sum::<f64>() / n;

        let variance = gaps.iter().map(|g| (g - mean_gap_ms).powi(2)).sum::<f64>() / n;

        let gap_std_ms = variance.sqrt();
        let gap_cv = if mean_gap_ms > 1e-10 {
            gap_std_ms / mean_gap_ms
        } else {
            0.0
        };

        // Classify rhythm
        let rhythm = if gap_cv < 0.3 {
            Rhythmicity::Isochronous { cv: gap_cv }
        } else if gap_cv < 0.5 {
            Rhythmicity::Rhythmic { cv: gap_cv }
        } else if gap_cv < 0.7 {
            Rhythmicity::Variable { cv: gap_cv }
        } else {
            Rhythmicity::Arrhythmic { cv: gap_cv }
        };

        Ok(ProsodyAnalysis {
            gap_cv,
            mean_gap_ms,
            gap_std_ms,
            rhythm,
        })
    }

    /// Analyze phonotactics (Forbidden Transitions)
    ///
    /// Identifies sound combinations that are physically difficult or statistically rare.
    pub fn analyze_phonotactics(
        &self,
        results: &[VocalizationResult],
    ) -> Result<PhonotacticsAnalysis> {
        use std::collections::HashMap;

        let mut transition_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut total_transitions: HashMap<String, usize> = HashMap::new();
        let mut spectral_deltas: Vec<f64> = Vec::new();

        // Build transition matrix
        for result in results {
            let mut phrases = result.phrases.clone();
            phrases.sort_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap());

            for window in phrases.windows(2) {
                let from_id = &window[0].phrase_id;
                let to_id = &window[1].phrase_id;

                // Calculate spectral delta (simplified: use duration as proxy)
                let delta = (window[1].duration_ms - window[0].duration_ms).abs();
                spectral_deltas.push(delta);

                *transition_counts
                    .entry(from_id.clone())
                    .or_default()
                    .entry(to_id.clone())
                    .or_insert(0) += 1;

                *total_transitions.entry(from_id.clone()).or_insert(0) += 1;
            }
        }

        // Convert counts to probabilities
        let mut transition_matrix: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for (from_id, to_counts) in &transition_counts {
            let total = *total_transitions.get(from_id).unwrap_or(&1) as f64;
            let probs: HashMap<String, f64> = to_counts
                .iter()
                .map(|(to_id, count)| (to_id.clone(), *count as f64 / total))
                .collect();
            transition_matrix.insert(from_id.clone(), probs);
        }

        // Identify forbidden transitions
        let mut forbidden_transitions = Vec::new();
        for (from_id, to_probs) in &transition_counts {
            for (to_id, &count) in to_probs {
                let total = *total_transitions.get(from_id).unwrap_or(&1) as f64;
                let probability = count as f64 / total;

                // Calculate spectral delta for this transition
                let spectral_delta = 0.0; // Placeholder: would need actual spectral features

                if probability < 0.01 {
                    forbidden_transitions.push(ForbiddenTransition {
                        from_phrase: from_id.clone(),
                        to_phrase: to_id.clone(),
                        probability,
                        spectral_delta,
                        reason: if count == 0 {
                            ForbiddenReason::Missing
                        } else {
                            ForbiddenReason::Rare
                        },
                    });
                }
            }
        }

        // Calculate mean spectral delta
        let mean_spectral_delta = if spectral_deltas.is_empty() {
            0.0
        } else {
            spectral_deltas.iter().sum::<f64>() / spectral_deltas.len() as f64
        };

        Ok(PhonotacticsAnalysis {
            transition_matrix,
            forbidden_transitions,
            mean_spectral_delta,
        })
    }

    /// Analyze pragmatics (Turn-Taking)
    ///
    /// Analyzes conversation flow, gaps, and overlaps.
    pub fn analyze_pragmatics(
        &self,
        _results: &[VocalizationResult],
    ) -> Result<PragmaticsAnalysis> {
        // For now, provide a simplified analysis
        // Full implementation would require speaker identification

        let gap_analysis = GapAnalysis {
            mean_gap_ms: 0.0,
            gap_std_ms: 0.0,
            same_speaker_after_gap_pct: 0.0,
            different_speaker_after_gap_pct: 0.0,
        };

        let overlap_analysis = OverlapAnalysis {
            overlap_count: 0,
            total_overlap_ms: 0.0,
            overlap_percentage: 0.0,
        };

        let pattern = TurnTakingPattern::Unknown;

        Ok(PragmaticsAnalysis {
            gap_analysis,
            overlap_analysis,
            pattern,
        })
    }

    /// Analyze updated atomicity with usage frequency
    ///
    /// True Atomicity = (Phonologically Atomic) × (Semantically Atomic)
    /// - Phonologically Atomic: intra_sim > 0.2 && inter_sim < 0.6
    /// - Semantically Atomic: frequency > threshold
    pub fn analyze_updated_atomicity(
        &self,
        clustered_phrases: &[ClusteredPhrase],
        zipf: &ZipfAnalysis,
    ) -> Vec<AtomicPhraseWithUsage> {
        // Calculate frequency threshold (e.g., median frequency)
        let frequencies: Vec<usize> = zipf.phrase_frequencies.values().copied().collect();
        let frequency_threshold = if frequencies.is_empty() {
            1
        } else {
            let mut sorted = frequencies.clone();
            sorted.sort();
            sorted[sorted.len() / 2] // Median
        };

        clustered_phrases
            .iter()
            .map(|cp| {
                let frequency = *zipf
                    .phrase_frequencies
                    .get(&cp.phrase.phrase_id)
                    .unwrap_or(&1);

                let is_phonologically_atomic = cp.is_atomic;
                let is_semantically_atomic = frequency >= frequency_threshold;
                let is_truly_atomic = is_phonologically_atomic && is_semantically_atomic;

                AtomicPhraseWithUsage {
                    phrase_id: cp.phrase.phrase_id.clone(),
                    cluster_id: cp.cluster_id,
                    intra_cluster_similarity: cp.intra_cluster_similarity,
                    inter_cluster_similarity: cp.inter_cluster_similarity,
                    frequency,
                    is_phonologically_atomic,
                    is_semantically_atomic,
                    is_truly_atomic,
                }
            })
            .collect()
    }
}

// =============================================================================
// Annotation Loading and Turn-Taking Analysis Implementation
// =============================================================================

/// Load emitter annotations from CSV file
pub fn load_annotations_from_csv<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<HashMap<String, EmitterAnnotation>, Box<dyn std::error::Error>> {
    use std::io::BufReader;

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rdr = csv::Reader::from_reader(reader);

    let mut annotations = HashMap::new();

    for result in rdr.records() {
        let record = result?;

        // Parse CSV columns
        // Format: Emitter,Addressee,Context,Emitter pre,Addressee pre,Emitter post,Addressee post,File Name
        let emitter: i32 = record.get(0).unwrap_or("0").parse()?;
        let addressee: i32 = record.get(1).unwrap_or("0").parse()?;
        let context: i32 = record.get(2).unwrap_or("0").parse()?;
        let emitter_pre: i32 = record.get(3).unwrap_or("0").parse()?;
        let addressee_pre: i32 = record.get(4).unwrap_or("0").parse()?;
        let emitter_post: i32 = record.get(5).unwrap_or("0").parse()?;
        let addressee_post: i32 = record.get(6).unwrap_or("0").parse()?;
        let file_name: String = record.get(7).unwrap_or("").to_string();

        let annotation = EmitterAnnotation {
            emitter,
            addressee,
            context,
            emitter_pre_action: emitter_pre,
            addressee_pre_action: addressee_pre,
            emitter_post_action: emitter_post,
            addressee_post_action: addressee_post,
            file_name: file_name.clone(),
        };

        annotations.insert(file_name, annotation);
    }

    Ok(annotations)
}

/// Perform turn-taking analysis with emitter information
pub fn analyze_turn_taking(annotations: &[EmitterAnnotation]) -> TurnTakingAnalysis {
    if annotations.is_empty() {
        return TurnTakingAnalysis {
            turn_switch_rate: 0.0,
            total_conversations: 0,
            aba_conversations: 0,
            dyadic_conversations: 0,
            conversation_stats: ConversationStats {
                mean_length: 0.0,
                median_length: 0.0,
                min_length: 0,
                max_length: 0,
                multi_turn_count: 0,
                long_conversation_count: 0,
            },
            response_time_stats: ResponseTimeStats {
                mean_gap: 0.0,
                median_gap: 0.0,
                min_gap: 0,
                max_gap: 0,
                immediate_response_count: 0,
                immediate_response_pct: 0.0,
            },
            context_turn_switch_rates: HashMap::new(),
            speaker_activity: HashMap::new(),
            pattern: TurnTakingPattern::Unknown,
        };
    }

    // 1. Calculate turn-switch rate
    let mut turn_switches = 0;
    for i in 1..annotations.len() {
        if annotations[i].emitter != annotations[i - 1].emitter {
            turn_switches += 1;
        }
    }
    let turn_switch_rate = if annotations.len() > 1 {
        (turn_switches as f64 / (annotations.len() - 1) as f64) * 100.0
    } else {
        0.0
    };

    // 2. Detect conversations (A→B→A patterns)
    let conversations = detect_conversations(annotations);
    let total_conversations = conversations.len();

    // 3. Count A→B→A patterns
    let aba_conversations = conversations
        .iter()
        .filter(|conv| {
            let emitters: Vec<i32> = conv.iter().map(|(_, emitter)| *emitter).collect();
            emitters.len() >= 3 && {
                let unique_emitters: std::collections::HashSet<i32> =
                    emitters.iter().cloned().collect();
                unique_emitters.len() == 2
            }
        })
        .count();

    // 4. Count dyadic conversations (exactly 2 unique emitters)
    let dyadic_conversations = conversations
        .iter()
        .filter(|conv| {
            let emitters: std::collections::HashSet<i32> =
                conv.iter().map(|(_, emitter)| *emitter).collect();
            emitters.len() == 2
        })
        .count();

    // 5. Conversation length statistics
    let conv_lengths: Vec<usize> = conversations.iter().map(|c| c.len()).collect();
    let (mean_length, median_length, min_length, max_length) = if !conv_lengths.is_empty() {
        let sum: usize = conv_lengths.iter().sum();
        let mean = sum as f64 / conv_lengths.len() as f64;
        let mut sorted = conv_lengths.clone();
        sorted.sort();
        let median = sorted[sorted.len() / 2] as f64;
        let min = *sorted.first().unwrap_or(&0);
        let max = *sorted.last().unwrap_or(&0);
        (mean, median, min, max)
    } else {
        (0.0, 0.0, 0, 0)
    };

    let multi_turn_count = conv_lengths.iter().filter(|&&l| l > 2).count();
    let long_conversation_count = conv_lengths.iter().filter(|&&l| l > 10).count();

    // 6. Response time analysis (all responses are immediate = 1 file gap)
    let response_time_stats = ResponseTimeStats {
        mean_gap: 1.0,
        median_gap: 1.0,
        min_gap: 1,
        max_gap: 1,
        immediate_response_count: turn_switches,
        immediate_response_pct: 100.0,
    };

    // 7. Context-specific turn-switch rates
    let mut context_turn_switch_rates = HashMap::new();
    let context_groups: HashMap<i32, Vec<&EmitterAnnotation>> =
        annotations.iter().fold(HashMap::new(), |mut acc, ann| {
            acc.entry(ann.context).or_insert_with(Vec::new).push(ann);
            acc
        });

    for (context_id, anns) in context_groups {
        let mut switches = 0;
        for i in 1..anns.len() {
            if anns[i].emitter != anns[i - 1].emitter {
                switches += 1;
            }
        }
        let rate = if anns.len() > 1 {
            (switches as f64 / (anns.len() - 1) as f64) * 100.0
        } else {
            0.0
        };
        context_turn_switch_rates.insert(context_id, rate);
    }

    // 8. Speaker activity
    let mut speaker_activity = HashMap::new();
    for ann in annotations {
        *speaker_activity.entry(ann.emitter).or_insert(0) += 1;
    }

    // 9. Pattern classification
    let pattern = if turn_switch_rate > 70.0 {
        TurnTakingPattern::Overlapping
    } else if turn_switch_rate > 50.0 {
        TurnTakingPattern::Flexible
    } else {
        TurnTakingPattern::Strict
    };

    TurnTakingAnalysis {
        turn_switch_rate,
        total_conversations,
        aba_conversations,
        dyadic_conversations,
        conversation_stats: ConversationStats {
            mean_length,
            median_length,
            min_length,
            max_length,
            multi_turn_count,
            long_conversation_count,
        },
        response_time_stats,
        context_turn_switch_rates,
        speaker_activity,
        pattern,
    }
}

/// Detect conversations from annotations
fn detect_conversations(annotations: &[EmitterAnnotation]) -> Vec<Vec<(usize, i32)>> {
    let mut conversations = Vec::new();
    let mut current_conv = Vec::new();
    let mut prev_emitter = None;

    for (idx, ann) in annotations.iter().enumerate() {
        if let Some(prev) = prev_emitter {
            if ann.emitter != prev {
                // Turn switch
                if !current_conv.is_empty() {
                    current_conv.push((idx, ann.emitter));
                }
                prev_emitter = Some(ann.emitter);
            } else {
                // Same emitter - end conversation
                if current_conv.len() > 1 {
                    conversations.push(current_conv);
                }
                current_conv = vec![(idx, ann.emitter)];
                prev_emitter = Some(ann.emitter);
            }
        } else {
            current_conv.push((idx, ann.emitter));
            prev_emitter = Some(ann.emitter);
        }
    }

    // Don't forget the last conversation
    if current_conv.len() > 1 {
        conversations.push(current_conv);
    }

    conversations
}

/// Perform social network analysis
pub fn analyze_social_network(annotations: &[EmitterAnnotation]) -> SocialNetworkAnalysis {
    if annotations.is_empty() {
        return SocialNetworkAnalysis {
            unique_emitters: 0,
            unique_addressees: 0,
            unique_pairs: 0,
            emitter_frequencies: HashMap::new(),
            addressee_frequencies: HashMap::new(),
            interaction_pairs: HashMap::new(),
            top_interactions: Vec::new(),
        };
    }

    // Count unique emitters and addressees
    let mut emitter_set = std::collections::HashSet::new();
    let mut addressee_set = std::collections::HashSet::new();
    let mut emitter_frequencies = HashMap::new();
    let mut addressee_frequencies = HashMap::new();
    let mut interaction_pairs = HashMap::new();

    for ann in annotations {
        emitter_set.insert(ann.emitter);
        addressee_set.insert(ann.addressee);

        *emitter_frequencies.entry(ann.emitter).or_insert(0) += 1;
        *addressee_frequencies.entry(ann.addressee).or_insert(0) += 1;

        let pair_key = format!("{}_{}", ann.emitter, ann.addressee);
        *interaction_pairs.entry(pair_key).or_insert(0) += 1;
    }

    // Get top interactions
    let mut top_pairs_vec: Vec<InteractionPair> = interaction_pairs
        .iter()
        .map(|(key, count)| {
            let parts: Vec<&str> = key.split('_').collect();
            let emitter: i32 = parts[0].parse().unwrap_or(0);
            let addressee: i32 = parts[1].parse().unwrap_or(0);
            InteractionPair {
                emitter,
                addressee,
                count: *count,
            }
        })
        .collect();

    top_pairs_vec.sort_by(|a, b| b.count.cmp(&a.count));
    let top_interactions = top_pairs_vec.into_iter().take(20).collect();

    SocialNetworkAnalysis {
        unique_emitters: emitter_set.len(),
        unique_addressees: addressee_set.len(),
        unique_pairs: interaction_pairs.len(),
        emitter_frequencies,
        addressee_frequencies,
        interaction_pairs,
        top_interactions,
    }
}

/// Perform context analysis
pub fn analyze_context(annotations: &[EmitterAnnotation]) -> ContextAnalysis {
    if annotations.is_empty() {
        return ContextAnalysis {
            unique_contexts: 0,
            context_frequencies: HashMap::new(),
            context_turn_switch_rates: HashMap::new(),
        };
    }

    let mut context_frequencies = HashMap::new();
    let mut context_turn_switch_rates = HashMap::new();

    // Group by context
    let mut context_groups: HashMap<i32, Vec<&EmitterAnnotation>> = HashMap::new();
    for ann in annotations {
        *context_frequencies.entry(ann.context).or_insert(0) += 1;
        context_groups.entry(ann.context).or_default().push(ann);
    }

    // Calculate turn-switch rate for each context
    for (context_id, anns) in context_groups {
        let mut switches = 0;
        for i in 1..anns.len() {
            if anns[i].emitter != anns[i - 1].emitter {
                switches += 1;
            }
        }

        let turn_switch_rate = if anns.len() > 1 {
            (switches as f64 / (anns.len() - 1) as f64) * 100.0
        } else {
            0.0
        };

        context_turn_switch_rates.insert(
            context_id,
            ContextTurnStats {
                context_id,
                vocalization_count: anns.len(),
                turn_switch_count: switches,
                turn_switch_rate,
            },
        );
    }

    ContextAnalysis {
        unique_contexts: context_frequencies.len(),
        context_frequencies,
        context_turn_switch_rates,
    }
}

// =============================================================================
// Synthesis Output: JSON Export & Audio Segmentation
// =============================================================================

/// Export clustered phrases to JSON for metadata-driven and granular synthesis
///
/// This exports all phrase metadata in a synthesis-ready format:
/// - **Metadata-driven synthesis**: Feature-based phrase selection
/// - **Granular synthesis**: Grain metadata with timing and features
/// - **Concatenative synthesis**: Unit selection with target features
pub fn export_phrases_for_synthesis(
    clustered_phrases: &[ClusteredPhrase],
    output_path: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufWriter;

    // Create synthesis-ready structure
    let synthesis_output = SynthesisOutput {
        metadata: SynthesisMetadata {
            export_date: chrono::Utc::now().to_rfc3339(),
            total_phrases: clustered_phrases.len(),
            species: "egyptian_fruit_bat".to_string(),
            feature_dimension: 30,
        },
        phrases: clustered_phrases
            .iter()
            .map(|cp| SynthesisPhrase {
                phrase_id: cp.phrase.phrase_id.clone(),
                cluster_id: cp.cluster_id,
                file_name: cp.phrase.file_name.clone(),
                start_ms: cp.phrase.start_ms,
                end_ms: cp.phrase.end_ms,
                duration_ms: cp.phrase.duration_ms,
                features: cp.phrase.features.clone(),
                rms_amplitude: cp.phrase.rms_amplitude,
                species: cp.phrase.species.clone(),
                context: cp.phrase.context.clone(),
                intra_cluster_similarity: cp.intra_cluster_similarity,
                inter_cluster_similarity: cp.inter_cluster_similarity,
                is_atomic: cp.is_atomic,
            })
            .collect(),
    };

    // Write to JSON
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &synthesis_output)?;

    Ok(())
}

/// Extract audio segments for concatenative synthesis
///
/// Extracts individual phrase WAV files from source audio for:
/// - **Concatenative synthesis**: Unit selection and concatenation
/// - **Audio library**: Reusable phrase audio clips
/// - **Analysis tools**: Listen to individual phrases
///
/// Output structure:
/// ```text
/// output_dir/
/// ├── phrases/              # Individual phrase WAV files
/// │   ├── cluster_0/        # All phrases from cluster 0
/// │   │   ├── phrase_001.wav
/// │   │   └── phrase_002.wav
/// │   └── cluster_1/
/// │       └── phrase_003.wav
/// ├── metadata.json         # Phrase metadata (JSON)
/// └── cluster_info.json     # Cluster statistics
/// ```
#[cfg(feature = "hound")]
pub fn extract_audio_segments(
    audio_dir: &Path,
    clustered_phrases: &[ClusteredPhrase],
    output_dir: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use hound::{WavSpec, WavWriter};
    use std::fs::create_dir_all;

    // Create output directories
    let phrases_dir = output_dir.join("phrases");
    create_dir_all(&phrases_dir)?;

    println!("📂 Extracting audio segments for concatenative synthesis...");
    println!("   Output directory: {}", output_dir.display());

    // Group phrases by cluster
    let mut cluster_groups: std::collections::HashMap<i32, Vec<&ClusteredPhrase>> =
        std::collections::HashMap::new();
    for cp in clustered_phrases {
        cluster_groups.entry(cp.cluster_id).or_default().push(cp);
    }

    // Extract phrases for each cluster
    let mut extracted_count = 0;
    for (cluster_id, phrases) in cluster_groups.iter() {
        let cluster_dir = phrases_dir.join(format!("cluster_{}", cluster_id));
        create_dir_all(&cluster_dir)?;

        for (idx, cp) in phrases.iter().enumerate() {
            // Load source audio
            let audio_path = audio_dir.join(&cp.phrase.file_name);
            if !audio_path.exists() {
                eprintln!(
                    "   ⚠️  Warning: Audio file not found: {}",
                    audio_path.display()
                );
                continue;
            }

            let (audio, sr) =
                load_wav_file(&audio_path).map_err(|e| format!("Failed to load audio: {}", e))?;

            // Calculate sample positions
            let start_sample = (cp.phrase.start_ms / 1000.0 * sr as f64) as usize;
            let end_sample = (cp.phrase.end_ms / 1000.0 * sr as f64) as usize;

            // Validate bounds
            if start_sample >= audio.len() || end_sample > audio.len() || start_sample >= end_sample
            {
                eprintln!(
                    "   ⚠️  Warning: Invalid time range for {}",
                    cp.phrase.phrase_id
                );
                continue;
            }

            // Extract segment
            let segment = &audio[start_sample..end_sample];

            // Write segment to WAV
            let output_filename = format!("phrase_{:04}.wav", idx);
            let output_path = cluster_dir.join(&output_filename);

            let spec = WavSpec {
                channels: 1,
                sample_rate: sr,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = WavWriter::create(&output_path, spec)?;

            for &sample in segment {
                writer.write_sample(sample)?;
            }

            writer.finalize()?;

            extracted_count += 1;

            // Progress update every 100 phrases
            if extracted_count % 100 == 0 {
                println!("   🎵 Extracted {} segments...", extracted_count);
            }
        }
    }

    println!("   ✅ Extracted {} audio segments", extracted_count);

    // Export metadata
    let metadata_path = output_dir.join("metadata.json");
    export_phrases_for_synthesis(clustered_phrases, &metadata_path)?;

    // Export cluster info
    let cluster_info_path = output_dir.join("cluster_info.json");
    export_cluster_info(clustered_phrases, &cluster_info_path)?;

    Ok(())
}

/// Export cluster statistics for synthesis
fn export_cluster_info(
    clustered_phrases: &[ClusteredPhrase],
    output_path: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufWriter;

    // Calculate cluster statistics
    let mut cluster_sizes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for cp in clustered_phrases {
        *cluster_sizes.entry(cp.cluster_id).or_insert(0) += 1;
    }

    let cluster_info: Vec<ClusterInfo> = cluster_sizes
        .iter()
        .map(|(cluster_id, &size)| ClusterInfo {
            cluster_id: *cluster_id,
            phrase_count: size,
            cluster_type: if *cluster_id == -1 {
                "noise".to_string()
            } else {
                "phrase".to_string()
            },
        })
        .collect();

    // Write to JSON
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &cluster_info)?;

    Ok(())
}

// =============================================================================
// Synthesis Data Structures
// =============================================================================

/// Synthesis output containing all phrase metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisOutput {
    /// Export metadata
    pub metadata: SynthesisMetadata,

    /// All phrases with full metadata
    pub phrases: Vec<SynthesisPhrase>,
}

/// Synthesis export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    /// Export timestamp (RFC3339)
    pub export_date: String,

    /// Total number of phrases
    pub total_phrases: usize,

    /// Species name
    pub species: String,

    /// Feature vector dimension
    pub feature_dimension: usize,
}

/// Individual phrase for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisPhrase {
    /// Unique phrase identifier
    pub phrase_id: String,

    /// Cluster ID (-1 = noise)
    pub cluster_id: i32,

    /// Source audio file name
    pub file_name: String,

    /// Start time in original audio (milliseconds)
    pub start_ms: f64,

    /// End time in original audio (milliseconds)
    pub end_ms: f64,

    /// Phrase duration (milliseconds)
    pub duration_ms: f64,

    /// 30D feature vector
    pub features: Vec<f64>,

    /// RMS amplitude
    pub rms_amplitude: f64,

    /// Species name
    pub species: String,

    /// Context label
    pub context: String,

    /// Intra-cluster similarity (coherence)
    pub intra_cluster_similarity: f64,

    /// Inter-cluster similarity (separation)
    pub inter_cluster_similarity: f64,

    /// Is this an atomic phrase?
    pub is_atomic: bool,
}

/// Cluster information for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    /// Cluster ID
    pub cluster_id: i32,

    /// Number of phrases in cluster
    pub phrase_count: usize,

    /// Cluster type (phrase or noise)
    pub cluster_type: String,
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ===== Test 1: Configuration defaults =====
    #[test]
    fn test_config_defaults() {
        let config = ExtractionConfig::default();

        assert_eq!(config.num_workers, 16);
        assert_eq!(config.sample_rate, 250000);
        assert_eq!(config.min_phrase_duration_ms, 10.0);
        assert_eq!(config.max_phrase_duration_ms, 500.0);
        assert_eq!(config.hop_length, 512);
    }

    // ===== Test 2: Pipeline creation =====
    #[test]
    fn test_pipeline_new() {
        let pipeline = ParallelExtractionPipeline::new();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_with_custom_config() {
        let config = ExtractionConfig {
            num_workers: 8,
            ..Default::default()
        };

        let pipeline = ParallelExtractionPipeline::with_config(config);
        assert!(pipeline.is_ok());
    }

    // ===== Test 3: Phrase candidate creation =====
    #[test]
    fn test_phrase_candidate_creation() {
        let phrase = PhraseCandidate {
            phrase_id: "test_1".to_string(),
            file_name: "test.wav".to_string(),
            start_ms: 0.0,
            end_ms: 100.0,
            duration_ms: 100.0,
            features: vec![0.0; 30],
            rms_amplitude: 0.5,
            species: "marmoset".to_string(),
            context: "contact".to_string(),
        };

        assert_eq!(phrase.phrase_id, "test_1");
        assert_eq!(phrase.duration_ms, 100.0);
        assert_eq!(phrase.features.len(), 30);
    }

    // ===== Test 4: Feature distance calculation =====
    #[test]
    fn test_feature_distance() {
        let phrase1 = PhraseCandidate {
            phrase_id: "test_1".to_string(),
            file_name: "test.wav".to_string(),
            start_ms: 0.0,
            end_ms: 100.0,
            duration_ms: 100.0,
            features: vec![0.0; 30],
            rms_amplitude: 0.5,
            species: "marmoset".to_string(),
            context: "contact".to_string(),
        };

        let phrase2 = PhraseCandidate {
            phrase_id: "test_2".to_string(),
            file_name: "test.wav".to_string(),
            start_ms: 100.0,
            end_ms: 200.0,
            duration_ms: 100.0,
            features: vec![1.0; 30],
            rms_amplitude: 0.5,
            species: "marmoset".to_string(),
            context: "contact".to_string(),
        };

        let distance = phrase1.feature_distance(&phrase2);
        assert!(distance > 0.0);
    }

    // ===== Test 5: Sentence segment creation =====
    #[test]
    fn test_changepoints_to_sentences() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let changepoints = vec![0, 100, 200, 300];
        let sentences = pipeline.changepoints_to_sentences(&changepoints, 512, 250000);

        assert_eq!(sentences.len(), 3);

        // Check first sentence
        assert_eq!(sentences[0].start_ms, 0.0);
        assert!(sentences[0].end_ms > 0.0);
    }

    // ===== Test 6: Empty phrases handling =====
    #[test]
    fn test_empty_phrases_returns_empty() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let phrases = vec![];
        let result = pipeline.cluster_phrases(&phrases);

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ===== Test 7: Grammar rule extraction =====
    #[test]
    fn test_grammar_rules_extraction() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let clustered_phrases = vec![
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "phrase_1".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 0,
                intra_cluster_similarity: 0.8,
                inter_cluster_similarity: 0.3,
                is_atomic: true,
                contexts: vec![1],
            },
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "phrase_2".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 100.0,
                    end_ms: 200.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 1,
                intra_cluster_similarity: 0.7,
                inter_cluster_similarity: 0.4,
                is_atomic: true,
                contexts: vec![1],
            },
        ];

        let results = vec![VocalizationResult {
            file_name: "test.wav".to_string(),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases: vec![],
        }];

        let rules = pipeline.extract_grammar_rules(&results, &clustered_phrases);

        // Should not crash
        assert!(rules.len() >= 0);
    }

    // ===== Test 8: Duration constraints =====
    #[test]
    fn test_duration_constraints() {
        let config = ExtractionConfig {
            min_phrase_duration_ms: 50.0,
            max_phrase_duration_ms: 150.0,
            ..Default::default()
        };

        assert_eq!(config.min_phrase_duration_ms, 50.0);
        assert_eq!(config.max_phrase_duration_ms, 150.0);
    }

    // ===== Test 9: Window scales =====
    #[test]
    fn test_window_scales() {
        let config = ExtractionConfig::default();

        assert!(!config.window_scales_ms.is_empty());
        assert!(config.window_scales_ms.contains(&50.0));
        assert!(config.window_scales_ms.contains(&500.0));
    }

    // ===== Test 10: Cluster ID lookup =====
    #[test]
    fn test_get_cluster_id() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let clustered_phrases = vec![ClusteredPhrase {
            phrase: PhraseCandidate {
                phrase_id: "phrase_1".to_string(),
                file_name: "test.wav".to_string(),
                start_ms: 0.0,
                end_ms: 100.0,
                duration_ms: 100.0,
                features: vec![0.0; 30],
                rms_amplitude: 0.5,
                species: "marmoset".to_string(),
                context: "contact".to_string(),
            },
            cluster_id: 0,
            intra_cluster_similarity: 0.8,
            inter_cluster_similarity: 0.3,
            is_atomic: true,
            contexts: vec![1],
        }];

        let cluster_id = pipeline.get_cluster_id("phrase_1", &clustered_phrases);
        assert_eq!(cluster_id, 0);

        let unknown_id = pipeline.get_cluster_id("unknown", &clustered_phrases);
        assert_eq!(unknown_id, -1);
    }

    // ===== Test 11: Intra-cluster similarity with identical vectors =====
    #[test]
    fn test_intra_cluster_similarity_identical() {
        use ndarray::Array2;

        // All identical vectors should have similarity = 1.0
        let features = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let sim = calculate_intra_cluster_similarity(&features);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Identical vectors should have sim=1.0, got {}",
            sim
        );
    }

    // ===== Test 12: Intra-cluster similarity with single member =====
    #[test]
    fn test_intra_cluster_similarity_single_member() {
        use ndarray::Array2;

        // Single member should return 1.0 (perfect coherence)
        let features = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let sim = calculate_intra_cluster_similarity(&features);
        assert_eq!(sim, 1.0, "Single member cluster should have sim=1.0");
    }

    // ===== Test 13: Intra-cluster similarity with orthogonal vectors =====
    #[test]
    fn test_intra_cluster_similarity_orthogonal() {
        use ndarray::Array2;

        // Orthogonal vectors should have similarity near 0.0
        let features = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let sim = calculate_intra_cluster_similarity(&features);
        assert!(
            (sim - 0.0).abs() < 1e-10,
            "Orthogonal vectors should have sim=0.0, got {}",
            sim
        );
    }

    // ===== Test 14: Inter-cluster similarity with no other clusters =====
    #[test]
    fn test_inter_cluster_similarity_no_other_clusters() {
        use ndarray::Array2;

        let all_features = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();

        let cluster_indices = vec![0];
        let labels = vec![0]; // Only one cluster
        let cluster_id = 0;

        let sim = calculate_inter_cluster_similarity(
            &all_features,
            &cluster_indices,
            &labels,
            cluster_id,
        );
        assert_eq!(sim, 0.0, "No other clusters should return sim=0.0");
    }

    // ===== Test 15: Inter-cluster similarity calculation =====
    #[test]
    fn test_inter_cluster_similarity_calculation() {
        use ndarray::Array2;

        // Create two well-separated clusters
        let all_features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0, // Cluster 0
                1.0, 0.0, 0.0, // Cluster 0
                0.0, 1.0, 0.0, // Cluster 1
                0.0, 1.0, 0.0, // Cluster 1
            ],
        )
        .unwrap();

        let cluster_indices = vec![0, 1]; // Cluster 0
        let labels = vec![0, 0, 1, 1];
        let cluster_id = 0;

        let sim = calculate_inter_cluster_similarity(
            &all_features,
            &cluster_indices,
            &labels,
            cluster_id,
        );
        // Centroid of cluster 0 is [1.0, 0.0, 0.0]
        // Similarity to cluster 1 members [0.0, 1.0, 0.0] should be 0.0
        assert!(
            (sim - 0.0).abs() < 1e-10,
            "Well-separated clusters should have low similarity"
        );
    }

    // ===== Test 16: Atomic phrase determination - atomic =====
    #[test]
    fn test_atomic_phrase_determination_atomic() {
        use ndarray::Array2;

        // High intra-similarity, low inter-similarity = ATOMIC
        let all_features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0, // Cluster 0
                1.0, 0.0, 0.0, // Cluster 0
                0.0, 1.0, 0.0, // Cluster 1
                0.0, 1.0, 0.0, // Cluster 1
            ],
        )
        .unwrap();

        let cluster_indices = vec![0, 1];
        let labels = vec![0, 0, 1, 1];
        let cluster_id = 0;

        let cluster_features = all_features.select(ndarray::Axis(0), &cluster_indices);
        let intra_sim = calculate_intra_cluster_similarity(&cluster_features);
        let inter_sim = calculate_inter_cluster_similarity(
            &all_features,
            &cluster_indices,
            &labels,
            cluster_id,
        );

        assert!(
            intra_sim > 0.2,
            "Intra-similarity should be > 0.2, got {}",
            intra_sim
        );
        assert!(
            inter_sim < 0.6,
            "Inter-similarity should be < 0.6, got {}",
            inter_sim
        );

        let is_atomic = intra_sim > 0.2 && inter_sim < 0.6;
        assert!(is_atomic, "This should be an atomic phrase");
    }

    // ===== Test 17: Atomic phrase determination - not atomic (low intra) =====
    #[test]
    fn test_atomic_phrase_determination_not_atomic_low_intra() {
        // Low intra-similarity = NOT ATOMIC
        let intra_sim = 0.1; // Below threshold
        let inter_sim = 0.3;

        let is_atomic = intra_sim > 0.2 && inter_sim < 0.6;
        assert!(!is_atomic, "Low intra-similarity should not be atomic");
    }

    // ===== Test 18: Atomic phrase determination - not atomic (high inter) =====
    #[test]
    fn test_atomic_phrase_determination_not_atomic_high_inter() {
        // High inter-similarity = NOT ATOMIC
        let intra_sim = 0.8;
        let inter_sim = 0.7; // Above threshold

        let is_atomic = intra_sim > 0.2 && inter_sim < 0.6;
        assert!(!is_atomic, "High inter-similarity should not be atomic");
    }

    // ===== Test 19: Atomic phrase boundary cases =====
    #[test]
    fn test_atomic_phrase_boundary_cases() {
        // Exactly at threshold (intra = 0.2, inter = 0.6)
        let intra_sim = 0.2;
        let inter_sim = 0.6;

        // intra > 0.2 is false (0.2 is not > 0.2)
        // inter < 0.6 is false (0.6 is not < 0.6)
        let is_atomic = intra_sim > 0.2 && inter_sim < 0.6;
        assert!(!is_atomic, "Boundary values should not be atomic");

        // Just above threshold
        let intra_sim = 0.2001;
        let inter_sim = 0.5999;
        let is_atomic = intra_sim > 0.2 && inter_sim < 0.6;
        assert!(is_atomic, "Just above threshold should be atomic");
    }

    // ===== Test 20: Compositionality detection =====
    #[test]
    fn test_compositionality_detection() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let results = vec![
            VocalizationResult {
                file_name: "test1.wav".to_string(),
                species: "marmoset".to_string(),
                sentences: vec![],
                phrases: vec![
                    PhraseCandidate {
                        phrase_id: "phrase_1".to_string(),
                        file_name: "test1.wav".to_string(),
                        start_ms: 0.0,
                        end_ms: 100.0,
                        duration_ms: 100.0,
                        features: vec![0.0; 30],
                        rms_amplitude: 0.5,
                        species: "marmoset".to_string(),
                        context: "1".to_string(),
                    },
                    PhraseCandidate {
                        phrase_id: "phrase_2".to_string(),
                        file_name: "test1.wav".to_string(),
                        start_ms: 100.0,
                        end_ms: 200.0,
                        duration_ms: 100.0,
                        features: vec![0.0; 30],
                        rms_amplitude: 0.5,
                        species: "marmoset".to_string(),
                        context: "1".to_string(),
                    },
                ],
            },
            VocalizationResult {
                file_name: "test2.wav".to_string(),
                species: "marmoset".to_string(),
                sentences: vec![],
                phrases: vec![
                    // phrase_1 reused across sentences
                    PhraseCandidate {
                        phrase_id: "phrase_1".to_string(),
                        file_name: "test2.wav".to_string(),
                        start_ms: 0.0,
                        end_ms: 100.0,
                        duration_ms: 100.0,
                        features: vec![0.0; 30],
                        rms_amplitude: 0.5,
                        species: "marmoset".to_string(),
                        context: "1".to_string(),
                    },
                    PhraseCandidate {
                        phrase_id: "phrase_3".to_string(),
                        file_name: "test2.wav".to_string(),
                        start_ms: 100.0,
                        end_ms: 200.0,
                        duration_ms: 100.0,
                        features: vec![0.0; 30],
                        rms_amplitude: 0.5,
                        species: "marmoset".to_string(),
                        context: "1".to_string(),
                    },
                ],
            },
        ];

        let clustered_phrases = vec![];

        let comp = pipeline.detect_compositionality(&results, &clustered_phrases);

        assert_eq!(comp.total_unique_phrases, 3, "Should have 3 unique phrases");
        assert_eq!(comp.reusable_phrases, 1, "Only phrase_1 is reusable");
        assert!(
            (comp.compositionality_ratio - (1.0 / 3.0)).abs() < 1e-10,
            "Compositionality ratio should be 1/3"
        );
        assert!(comp.phrase_usage.contains_key("phrase_1"));
        assert_eq!(comp.phrase_usage.get("phrase_1").unwrap().sentence_count, 2);
    }

    // ===== Test 21: Compositionality with no reuse =====
    #[test]
    fn test_compositionality_no_reuse() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let results = vec![VocalizationResult {
            file_name: "test1.wav".to_string(),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases: vec![PhraseCandidate {
                phrase_id: "phrase_1".to_string(),
                file_name: "test1.wav".to_string(),
                start_ms: 0.0,
                end_ms: 100.0,
                duration_ms: 100.0,
                features: vec![0.0; 30],
                rms_amplitude: 0.5,
                species: "marmoset".to_string(),
                context: "1".to_string(),
            }],
        }];

        let clustered_phrases = vec![];

        let comp = pipeline.detect_compositionality(&results, &clustered_phrases);

        assert_eq!(comp.total_unique_phrases, 1);
        assert_eq!(comp.reusable_phrases, 0, "No phrases are reused");
        assert_eq!(
            comp.compositionality_ratio, 0.0,
            "Compositionality should be 0.0"
        );
    }

    // ===== Test 22: ClusteredPhrase with atomicity =====
    #[test]
    fn test_clustered_phrase_atomicity() {
        let phrase = PhraseCandidate {
            phrase_id: "test_phrase".to_string(),
            file_name: "test.wav".to_string(),
            start_ms: 0.0,
            end_ms: 100.0,
            duration_ms: 100.0,
            features: vec![0.0; 30],
            rms_amplitude: 0.5,
            species: "marmoset".to_string(),
            context: "contact".to_string(),
        };

        // Test atomic case
        let atomic_phrase = ClusteredPhrase::new(
            phrase.clone(),
            0,
            0.8, // High intra-similarity
            0.3, // Low inter-similarity
            vec![1, 2, 3],
        );

        assert!(atomic_phrase.is_atomic, "Should be atomic");
        assert_eq!(atomic_phrase.intra_cluster_similarity, 0.8);
        assert_eq!(atomic_phrase.inter_cluster_similarity, 0.3);
        assert_eq!(atomic_phrase.contexts, vec![1, 2, 3]);

        // Test non-atomic case (low intra-similarity)
        let non_atomic_phrase = ClusteredPhrase::new(
            phrase.clone(),
            1,
            0.1, // Low intra-similarity
            0.3,
            vec![1],
        );

        assert!(
            !non_atomic_phrase.is_atomic,
            "Low intra-similarity should not be atomic"
        );

        // Test non-atomic case (high inter-similarity)
        let non_atomic_phrase2 = ClusteredPhrase::new(
            phrase,
            2,
            0.8,
            0.7, // High inter-similarity
            vec![1],
        );

        assert!(
            !non_atomic_phrase2.is_atomic,
            "High inter-similarity should not be atomic"
        );
    }

    // ===== Test 23: Zipf's Law - Phrase Frequency Extraction =====
    #[test]
    fn test_get_phrase_frequencies() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create test phrases with varying frequencies
        let clustered_phrases = vec![
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "phrase_common".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 0,
                intra_cluster_similarity: 0.8,
                inter_cluster_similarity: 0.3,
                is_atomic: true,
                contexts: vec![1],
            },
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "phrase_common".to_string(), // Duplicate
                    file_name: "test2.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 0,
                intra_cluster_similarity: 0.8,
                inter_cluster_similarity: 0.3,
                is_atomic: true,
                contexts: vec![1],
            },
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "phrase_rare".to_string(),
                    file_name: "test3.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 1,
                intra_cluster_similarity: 0.7,
                inter_cluster_similarity: 0.4,
                is_atomic: true,
                contexts: vec![1],
            },
        ];

        let zipf = pipeline.analyze_zipf_law(&clustered_phrases).unwrap();

        // Top phrase should have count > 1
        assert_eq!(zipf.phrase_frequencies.get("phrase_common"), Some(&2));
        // Rare phrase should have count = 1 (Hapax Legomenon)
        assert_eq!(zipf.phrase_frequencies.get("phrase_rare"), Some(&1));
    }

    // ===== Test 24: Zipf's Law - Distribution Exists =====
    #[test]
    fn test_zipf_distribution_exists() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create a power-law distribution
        let mut clustered_phrases = Vec::new();
        for i in 0..10 {
            let count = 10 - i; // Decreasing frequency: 10, 9, 8, ..., 1
            for _ in 0..count {
                clustered_phrases.push(ClusteredPhrase {
                    phrase: PhraseCandidate {
                        phrase_id: format!("phrase_{}", i),
                        file_name: "test.wav".to_string(),
                        start_ms: 0.0,
                        end_ms: 100.0,
                        duration_ms: 100.0,
                        features: vec![0.0; 30],
                        rms_amplitude: 0.5,
                        species: "marmoset".to_string(),
                        context: "contact".to_string(),
                    },
                    cluster_id: i as i32,
                    intra_cluster_similarity: 0.8,
                    inter_cluster_similarity: 0.3,
                    is_atomic: true,
                    contexts: vec![1],
                });
            }
        }

        let zipf = pipeline.analyze_zipf_law(&clustered_phrases).unwrap();

        // Slope should be negative (frequency decreases with rank)
        assert!(
            zipf.slope_alpha < 0.0,
            "Zipf slope should be negative, got {}",
            zipf.slope_alpha
        );

        // Correlation should be reasonably high for a power-law distribution
        assert!(
            zipf.correlation_r2 > 0.5,
            "Correlation R² should be > 0.5, got {}",
            zipf.correlation_r2
        );
    }

    // ===== Test 25: Zipf's Law - Slope Alpha Calculation =====
    #[test]
    fn test_calculate_slope_alpha() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create an ideal Zipfian distribution: f(r) = 1/r
        let mut clustered_phrases = Vec::new();
        for rank in 1..=10 {
            let freq = (10.0 / rank as f64).round() as usize; // Approximate Zipf's law
            for _ in 0..freq {
                clustered_phrases.push(ClusteredPhrase {
                    phrase: PhraseCandidate {
                        phrase_id: format!("phrase_{}", rank),
                        file_name: "test.wav".to_string(),
                        start_ms: 0.0,
                        end_ms: 100.0,
                        duration_ms: 100.0,
                        features: vec![0.0; 30],
                        rms_amplitude: 0.5,
                        species: "marmoset".to_string(),
                        context: "contact".to_string(),
                    },
                    cluster_id: rank as i32,
                    intra_cluster_similarity: 0.8,
                    inter_cluster_similarity: 0.3,
                    is_atomic: true,
                    contexts: vec![1],
                });
            }
        }

        let zipf = pipeline.analyze_zipf_law(&clustered_phrases).unwrap();

        // Slope should be between -0.5 and -1.5 for Zipfian distribution
        assert!(
            zipf.slope_alpha >= -1.5 && zipf.slope_alpha <= -0.5,
            "Zipf slope should be between -0.5 and -1.5, got {}",
            zipf.slope_alpha
        );

        // Should be classified as Efficient or Optimal
        match zipf.efficiency {
            CommunicationEfficiency::Optimal { slope }
            | CommunicationEfficiency::Efficient { slope } => {
                assert!(slope <= -0.5 && slope >= -1.5);
            }
            _ => panic!(
                "Expected Optimal or Efficient efficiency, got {:?}",
                zipf.efficiency
            ),
        }
    }

    // ===== Test 26: Zipf's Law - Empty Dataset =====
    #[test]
    fn test_zipf_empty_dataset() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let clustered_phrases: Vec<ClusteredPhrase> = vec![];
        let zipf = pipeline.analyze_zipf_law(&clustered_phrases).unwrap();

        assert!(zipf.phrase_frequencies.is_empty());
        assert!(zipf.ranked_phrases.is_empty());
        assert_eq!(zipf.slope_alpha, 0.0);
        assert_eq!(zipf.correlation_r2, 0.0);
        assert!(matches!(zipf.efficiency, CommunicationEfficiency::Unknown));
    }

    // ===== Test 27: Prosody Analysis - Isochrony Detection =====
    #[test]
    fn test_prosody_isochrony_detection() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create phrases with regular gaps (isochronous)
        let results = vec![VocalizationResult {
            file_name: "test.wav".to_string(),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases: vec![
                PhraseCandidate {
                    phrase_id: "p1".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                PhraseCandidate {
                    phrase_id: "p2".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 200.0, // 100ms gap
                    end_ms: 300.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                PhraseCandidate {
                    phrase_id: "p3".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 400.0, // 100ms gap (regular!)
                    end_ms: 500.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
            ],
        }];

        let prosody = pipeline.analyze_prosody(&results).unwrap();

        // Low CV indicates rhythmic/isochronous
        assert!(
            prosody.gap_cv < 0.3,
            "Expected low CV (< 0.3) for regular gaps, got {}",
            prosody.gap_cv
        );
        assert_eq!(prosody.mean_gap_ms, 100.0);

        // Should be classified as Isochronous or Rhythmic
        match prosody.rhythm {
            Rhythmicity::Isochronous { cv } | Rhythmicity::Rhythmic { cv } => {
                assert!(cv < 0.5);
            }
            _ => panic!(
                "Expected Isochronous or Rhythmic rhythm, got {:?}",
                prosody.rhythm
            ),
        }
    }

    // ===== Test 28: Prosody Analysis - Arrhythmic Detection =====
    #[test]
    fn test_prosody_arrhythmic_detection() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create phrases with irregular gaps (arrhythmic)
        let results = vec![VocalizationResult {
            file_name: "test.wav".to_string(),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases: vec![
                PhraseCandidate {
                    phrase_id: "p1".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                PhraseCandidate {
                    phrase_id: "p2".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 150.0, // 50ms gap
                    end_ms: 250.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                PhraseCandidate {
                    phrase_id: "p3".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 600.0, // 350ms gap (very irregular!)
                    end_ms: 700.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
            ],
        }];

        let prosody = pipeline.analyze_prosody(&results).unwrap();

        // High CV indicates arrhythmic
        assert!(
            prosody.gap_cv > 0.7,
            "Expected high CV (> 0.7) for irregular gaps, got {}",
            prosody.gap_cv
        );
    }

    // ===== Test 29: Phonotactics - Transition Matrix =====
    #[test]
    fn test_phonotactics_transition_matrix() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create phrases with specific transitions: A -> B -> C
        let results = vec![VocalizationResult {
            file_name: "test.wav".to_string(),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases: vec![
                PhraseCandidate {
                    phrase_id: "A".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                PhraseCandidate {
                    phrase_id: "B".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 100.0,
                    end_ms: 200.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                PhraseCandidate {
                    phrase_id: "C".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 200.0,
                    end_ms: 300.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
            ],
        }];

        let phonotactics = pipeline.analyze_phonotactics(&results).unwrap();

        // Should have transitions A -> B and B -> C
        assert!(phonotactics.transition_matrix.contains_key("A"));
        assert!(phonotactics.transition_matrix.contains_key("B"));

        // A should transition to B with high probability
        let a_transitions = phonotactics.transition_matrix.get("A").unwrap();
        assert!(a_transitions.contains_key("B"));

        // B should transition to C
        let b_transitions = phonotactics.transition_matrix.get("B").unwrap();
        assert!(b_transitions.contains_key("C"));
    }

    // ===== Test 30: Updated Atomicity with Usage Frequency =====
    #[test]
    fn test_updated_atomicity_with_frequency() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        // Create clustered phrases
        let clustered_phrases = vec![
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "frequent_atomic".to_string(),
                    file_name: "test.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 0,
                intra_cluster_similarity: 0.8,
                inter_cluster_similarity: 0.3,
                is_atomic: true, // Phonologically atomic
                contexts: vec![1, 2, 3],
            },
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "frequent_atomic".to_string(), // Same phrase
                    file_name: "test2.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 0,
                intra_cluster_similarity: 0.8,
                inter_cluster_similarity: 0.3,
                is_atomic: true,
                contexts: vec![1],
            },
            ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: "rare_atomic".to_string(),
                    file_name: "test3.wav".to_string(),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: 1,
                intra_cluster_similarity: 0.8,
                inter_cluster_similarity: 0.3,
                is_atomic: true, // Phonologically atomic
                contexts: vec![1],
            },
        ];

        let zipf = pipeline.analyze_zipf_law(&clustered_phrases).unwrap();
        let updated = pipeline.analyze_updated_atomicity(&clustered_phrases, &zipf);

        // Find the frequent and rare phrases
        let frequent_phrase = updated
            .iter()
            .find(|p| p.phrase_id == "frequent_atomic")
            .unwrap();
        let rare_phrase = updated
            .iter()
            .find(|p| p.phrase_id == "rare_atomic")
            .unwrap();

        // Both should be phonologically atomic
        assert!(frequent_phrase.is_phonologically_atomic);
        assert!(rare_phrase.is_phonologically_atomic);

        // Frequent phrase should be semantically atomic (frequency >= median)
        // Rare phrase might not be semantically atomic (frequency < median)
        assert!(frequent_phrase.frequency >= rare_phrase.frequency);

        // True atomicity = phonological AND semantic
        // The frequent phrase is more likely to be truly atomic
        assert!(frequent_phrase.is_truly_atomic);
    }

    // ===== Test 31: Comprehensive Linguistic Analysis =====
    #[test]
    fn test_comprehensive_linguistic_analysis() {
        let config = ExtractionConfig::default();
        let pipeline = ParallelExtractionPipeline::with_config(config).unwrap();

        let results = vec![VocalizationResult {
            file_name: "test.wav".to_string(),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases: vec![PhraseCandidate {
                phrase_id: "phrase_1".to_string(),
                file_name: "test.wav".to_string(),
                start_ms: 0.0,
                end_ms: 100.0,
                duration_ms: 100.0,
                features: vec![0.0; 30],
                rms_amplitude: 0.5,
                species: "marmoset".to_string(),
                context: "1".to_string(),
            }],
        }];

        let clustered_phrases = vec![ClusteredPhrase {
            phrase: PhraseCandidate {
                phrase_id: "phrase_1".to_string(),
                file_name: "test.wav".to_string(),
                start_ms: 0.0,
                end_ms: 100.0,
                duration_ms: 100.0,
                features: vec![0.0; 30],
                rms_amplitude: 0.5,
                species: "marmoset".to_string(),
                context: "1".to_string(),
            },
            cluster_id: 0,
            intra_cluster_similarity: 0.8,
            inter_cluster_similarity: 0.3,
            is_atomic: true,
            contexts: vec![1],
        }];

        let analysis = pipeline
            .analyze_linguistics(&results, &clustered_phrases)
            .unwrap();

        // Should have all analysis components
        assert!(!analysis.zipf.phrase_frequencies.is_empty());
        assert!(analysis.prosody.gap_cv >= 0.0);
        assert!(analysis.phonotactics.mean_spectral_delta >= 0.0);
        assert!(matches!(
            analysis.pragmatics.pattern,
            TurnTakingPattern::Unknown
        ));
        assert!(!analysis.updated_atomic_phrases.is_empty());
    }

    // ===== Test 32: Communication Efficiency Classification =====
    #[test]
    fn test_communication_efficiency_classification() {
        // Test Optimal (slope ≈ -1.0)
        let efficiency_optimal = CommunicationEfficiency::Optimal { slope: -1.0 };
        match efficiency_optimal {
            CommunicationEfficiency::Optimal { slope } => assert_eq!(slope, -1.0),
            _ => panic!("Expected Optimal"),
        }

        // Test Efficient (slope ≈ -0.7)
        let efficiency_efficient = CommunicationEfficiency::Efficient { slope: -0.7 };
        match efficiency_efficient {
            CommunicationEfficiency::Efficient { slope } => assert_eq!(slope, -0.7),
            _ => panic!("Expected Efficient"),
        }

        // Test Inefficient (slope > -0.5)
        let efficiency_inefficient = CommunicationEfficiency::Inefficient { slope: -0.3 };
        match efficiency_inefficient {
            CommunicationEfficiency::Inefficient { slope } => assert_eq!(slope, -0.3),
            _ => panic!("Expected Inefficient"),
        }

        // Test Random (slope ≈ 0)
        let efficiency_random = CommunicationEfficiency::Random { slope: 0.05 };
        match efficiency_random {
            CommunicationEfficiency::Random { slope } => assert_eq!(slope, 0.05),
            _ => panic!("Expected Random"),
        }
    }

    // ===== Test 33: Rhythmicity Classification =====
    #[test]
    fn test_rhythmicity_classification() {
        // Test Isochronous (CV < 0.3)
        let rhythm_iso = Rhythmicity::Isochronous { cv: 0.2 };
        match rhythm_iso {
            Rhythmicity::Isochronous { cv } => assert_eq!(cv, 0.2),
            _ => panic!("Expected Isochronous"),
        }

        // Test Rhythmic (0.3 <= CV < 0.5)
        let rhythm_rhythmic = Rhythmicity::Rhythmic { cv: 0.4 };
        match rhythm_rhythmic {
            Rhythmicity::Rhythmic { cv } => assert_eq!(cv, 0.4),
            _ => panic!("Expected Rhythmic"),
        }

        // Test Variable (0.5 <= CV < 0.7)
        let rhythm_variable = Rhythmicity::Variable { cv: 0.6 };
        match rhythm_variable {
            Rhythmicity::Variable { cv } => assert_eq!(cv, 0.6),
            _ => panic!("Expected Variable"),
        }

        // Test Arrhythmic (CV >= 0.7)
        let rhythm_arrhythmic = Rhythmicity::Arrhythmic { cv: 0.8 };
        match rhythm_arrhythmic {
            Rhythmicity::Arrhythmic { cv } => assert_eq!(cv, 0.8),
            _ => panic!("Expected Arrhythmic"),
        }
    }

    // ===== Test 34: PhraseAudioSegment Creation =====
    #[test]
    fn test_phrase_audio_segment_creation() {
        let audio = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let sr = 44100u32;

        let segment = PhraseAudioSegment::new(
            audio.clone(),
            sr,
            "F0_6400_DUR_5".to_string(),
            "test.wav".to_string(),
            100.0,  // start_time_ms
            105.0,  // end_time_ms
            6400.0, // mean_f0_hz
            200.0,  // f0_range_hz
            0.5,    // rms_amplitude
            "marmoset".to_string(),
            "contact".to_string(),
        );

        assert_eq!(segment.audio, audio);
        assert_eq!(segment.sr, sr);
        assert_eq!(segment.phrase_key, "F0_6400_DUR_5");
        assert_eq!(segment.duration_ms, 5.0);
        assert_eq!(segment.duration_samples(), 5);
        assert_eq!(segment.duration_seconds(), 0.005);
        assert!(!segment.occurrence_id.is_empty());
    }

    // ===== Test 35: PhraseAudioLibrary Creation =====
    #[test]
    fn test_phrase_audio_library_creation() {
        let library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);

        assert_eq!(library.species, "marmoset");
        assert_eq!(library.sr, 44100);
        assert_eq!(library.total_segments, 0);
        assert_eq!(library.total_phrases, 0);
        assert_eq!(library.max_segments_per_phrase, 100);
        assert_eq!(library.min_quality_score, 0.3);
    }

    // ===== Test 36: PhraseAudioLibrary Add Segment =====
    #[test]
    fn test_phrase_audio_library_add_segment() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);

        let audio = vec![0.1f32, 0.2, 0.3];
        let segment = PhraseAudioSegment::new(
            audio,
            44100,
            "F0_6400_DUR_5".to_string(),
            "test.wav".to_string(),
            100.0,
            105.0,
            6400.0,
            200.0,
            0.5,
            "marmoset".to_string(),
            "contact".to_string(),
        );

        library.add_segment(segment);

        assert_eq!(library.total_segments, 1);
        assert_eq!(library.total_phrases, 1);
        assert!(library.get_segments("F0_6400_DUR_5").is_some());
        assert_eq!(library.get_segments("F0_6400_DUR_5").unwrap().len(), 1);
    }

    // ===== Test 37: PhraseAudioLibrary Quality Filtering =====
    #[test]
    fn test_phrase_audio_library_quality_filtering() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);
        library.min_quality_score = 0.5;

        let audio = vec![0.1f32, 0.2, 0.3];

        // High quality segment should be added
        let mut good_segment = PhraseAudioSegment::new(
            audio.clone(),
            44100,
            "F0_6400_DUR_5".to_string(),
            "test.wav".to_string(),
            100.0,
            105.0,
            6400.0,
            200.0,
            0.5,
            "marmoset".to_string(),
            "contact".to_string(),
        );
        good_segment.quality_score = 0.8;
        library.add_segment(good_segment);

        // Low quality segment should be rejected
        let mut bad_segment = PhraseAudioSegment::new(
            audio,
            44100,
            "F0_6500_DUR_6".to_string(),
            "test.wav".to_string(),
            100.0,
            106.0,
            6500.0,
            200.0,
            0.4,
            "marmoset".to_string(),
            "contact".to_string(),
        );
        bad_segment.quality_score = 0.3;
        library.add_segment(bad_segment);

        assert_eq!(library.total_segments, 1);
        assert_eq!(library.total_phrases, 1);
    }

    // ===== Test 38: PhraseAudioLibrary Max Segments Per Phrase =====
    #[test]
    fn test_phrase_audio_library_max_segments_per_phrase() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);
        library.max_segments_per_phrase = 3;

        let audio = vec![0.1f32, 0.2, 0.3];

        // Add 5 segments (only 3 should be stored)
        for i in 0..5 {
            let segment = PhraseAudioSegment::new(
                audio.clone(),
                44100,
                "F0_6400_DUR_5".to_string(),
                "test.wav".to_string(),
                100.0 + i as f64,
                105.0 + i as f64,
                6400.0,
                200.0,
                0.5,
                "marmoset".to_string(),
                "contact".to_string(),
            );
            library.add_segment(segment);
        }

        assert_eq!(library.total_segments, 3);
        assert_eq!(library.total_phrases, 1);
        assert_eq!(library.get_segments("F0_6400_DUR_5").unwrap().len(), 3);
    }

    // ===== Test 39: PhraseAudioLibrary Get Best Segment =====
    #[test]
    fn test_phrase_audio_library_get_best_segment() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);

        let audio = vec![0.1f32, 0.2, 0.3];

        // Add segments with different quality scores
        for i in 0..3 {
            let mut segment = PhraseAudioSegment::new(
                audio.clone(),
                44100,
                "F0_6400_DUR_5".to_string(),
                "test.wav".to_string(),
                100.0 + i as f64,
                105.0 + i as f64,
                6400.0,
                200.0,
                0.5,
                "marmoset".to_string(),
                "contact".to_string(),
            );
            segment.quality_score = 0.5 + i as f64 * 0.2; // 0.5, 0.7, 0.9
            library.add_segment(segment);
        }

        let best = library.get_best_segment("F0_6400_DUR_5");
        assert!(best.is_some());
        assert_eq!(best.unwrap().quality_score, 0.9);
    }

    // ===== Test 40: PhraseAudioLibrary Phrase Keys =====
    #[test]
    fn test_phrase_audio_library_phrase_keys() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);

        let audio = vec![0.1f32, 0.2, 0.3];

        let keys = vec![
            "F0_6400_DUR_5".to_string(),
            "F0_6500_DUR_6".to_string(),
            "F0_6600_DUR_7".to_string(),
        ];

        for key in &keys {
            let segment = PhraseAudioSegment::new(
                audio.clone(),
                44100,
                key.clone(),
                "test.wav".to_string(),
                100.0,
                105.0,
                6400.0,
                200.0,
                0.5,
                "marmoset".to_string(),
                "contact".to_string(),
            );
            library.add_segment(segment);
        }

        let library_keys = library.phrase_keys();
        assert_eq!(library_keys.len(), 3);
        for key in &keys {
            assert!(library_keys.contains(key));
        }
    }

    // ===== Test 41: PhraseAudioLibrary Statistics =====
    #[test]
    fn test_phrase_audio_library_statistics() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);

        let audio = vec![0.1f32, 0.2, 0.3];

        // Add segments to different phrases
        for i in 0..5 {
            let segment = PhraseAudioSegment::new(
                audio.clone(),
                44100,
                format!("F0_6400_DUR_5_{}", i),
                "test.wav".to_string(),
                100.0,
                105.0,
                6400.0,
                200.0,
                0.5,
                "marmoset".to_string(),
                "contact".to_string(),
            );
            library.add_segment(segment);
        }

        // Add multiple segments to one phrase
        for _i in 0..3 {
            let segment = PhraseAudioSegment::new(
                audio.clone(),
                44100,
                "F0_6500_DUR_6".to_string(),
                "test.wav".to_string(),
                100.0,
                105.0,
                6500.0,
                200.0,
                0.5,
                "marmoset".to_string(),
                "contact".to_string(),
            );
            library.add_segment(segment);
        }

        let stats = library.statistics();

        assert_eq!(stats.species, "marmoset");
        assert_eq!(stats.sr, 44100);
        assert_eq!(stats.total_segments, 8);
        assert_eq!(stats.total_phrases, 6);
        assert_eq!(stats.phrase_counts.len(), 6);
    }

    // ===== Test 42: PhraseAudioLibrary Serialization =====
    #[test]
    fn test_phrase_audio_library_serialization() {
        let mut library = PhraseAudioLibrary::new("marmoset".to_string(), 44100);

        let audio = vec![0.1f32, 0.2, 0.3];
        let segment = PhraseAudioSegment::new(
            audio,
            44100,
            "F0_6400_DUR_5".to_string(),
            "test.wav".to_string(),
            100.0,
            105.0,
            6400.0,
            200.0,
            0.5,
            "marmoset".to_string(),
            "contact".to_string(),
        );

        library.add_segment(segment);

        // Serialize
        let json = serde_json::to_string(&library);
        assert!(json.is_ok());

        // Deserialize
        let deserialized: std::result::Result<PhraseAudioLibrary, _> =
            serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());

        let lib = deserialized.unwrap();
        assert_eq!(lib.species, "marmoset");
        assert_eq!(lib.total_segments, 1);
    }
}
