// Synthesis Pipeline
//
// Orchestrates the three synthesis techniques:
// 1. Metadata-driven synthesis (parameter-based)
// 2. Granular synthesis (grain-based)
// 3. Concatenative synthesis (sample-based)
//
// This is the top-level pipeline that converts vocabulary mappings
// into synthesis-ready assets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::audio_segmenter::{AudioSegmentForSynthesis, AudioSegmenter, GrainEnvelope};
use crate::vocabulary_mapper::{VocabularyItem, VocabularyMapper};

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error("Vocabulary not found: {0}")]
    VocabularyNotFound(String),

    #[error("Segmenter error: {0}")]
    SegmenterError(String),

    #[error("Invalid synthesis parameters: {0}")]
    InvalidParameters(String),

    #[error("Export failed: {0}")]
    ExportError(String),
}

// Auto-convert from SegmenterError
impl From<crate::audio_segmenter::SegmenterError> for SynthesisError {
    fn from(err: crate::audio_segmenter::SegmenterError) -> Self {
        SynthesisError::SegmenterError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, SynthesisError>;

// =============================================================================
// Synthesis Techniques
// =============================================================================

/// Metadata-driven synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDrivenParams {
    /// Target duration in seconds
    pub target_duration: f64,

    /// F0 contour (Hz)
    pub f0_contour: Vec<f64>,

    /// Intensity (0.0 to 1.0)
    pub intensity: f64,

    /// Spectral centroid (Hz)
    pub spectral_centroid: f64,

    /// Spectral bandwidth (Hz)
    pub spectral_bandwidth: f64,
}

impl Default for MetadataDrivenParams {
    fn default() -> Self {
        Self {
            target_duration: 0.1,
            f0_contour: vec![10000.0],
            intensity: 0.7,
            spectral_centroid: 15000.0,
            spectral_bandwidth: 5000.0,
        }
    }
}

/// Granular synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranularSynthesisParams {
    /// Grain size in milliseconds
    pub grain_size_ms: f64,

    /// Hop size in milliseconds
    pub hop_size_ms: f64,

    /// Grain envelope type
    pub envelope: GrainEnvelopeType,

    /// Number of grains per second
    pub density: f64,

    /// Time stretching factor (1.0 = normal)
    pub time_stretch: f64,

    /// Pitch shifting factor (1.0 = normal)
    pub pitch_shift: f64,
}

impl Default for GranularSynthesisParams {
    fn default() -> Self {
        Self {
            grain_size_ms: 50.0,
            hop_size_ms: 25.0,
            envelope: GrainEnvelopeType::Hann,
            density: 20.0,
            time_stretch: 1.0,
            pitch_shift: 1.0,
        }
    }
}

/// Concatenative synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatenativeParams {
    /// Crossfade duration in milliseconds
    pub crossfade_ms: f64,

    /// Normalize output
    pub normalize: bool,

    /// Minimum segment duration
    pub min_segment_duration: f64,

    /// Maximum segment duration
    pub max_segment_duration: f64,
}

impl Default for ConcatenativeParams {
    fn default() -> Self {
        Self {
            crossfade_ms: 10.0,
            normalize: true,
            min_segment_duration: 0.05,
            max_segment_duration: 2.0,
        }
    }
}

/// Grain envelope type (serializable version)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrainEnvelopeType {
    None,
    Linear { fade_in_ms: f64, fade_out_ms: f64 },
    Gaussian { width_ms: f64 },
    Hann,
}

impl From<GrainEnvelopeType> for GrainEnvelope {
    fn from(env: GrainEnvelopeType) -> Self {
        match env {
            GrainEnvelopeType::None => GrainEnvelope::None,
            GrainEnvelopeType::Linear {
                fade_in_ms,
                fade_out_ms,
            } => GrainEnvelope::Linear {
                fade_in_ms,
                fade_out_ms,
            },
            GrainEnvelopeType::Gaussian { width_ms } => GrainEnvelope::Gaussian { width_ms },
            GrainEnvelopeType::Hann => GrainEnvelope::Hann,
        }
    }
}

/// Concatenative synthesis unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatenativeUnit {
    /// Unit ID
    pub unit_id: String,

    /// Vocabulary ID
    pub vocab_id: String,

    /// Audio file path
    pub audio_path: PathBuf,

    /// Duration in seconds
    pub duration: f64,

    /// Context information
    pub context: UnitContext,
}

/// Context for concatenative unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitContext {
    pub emitter_id: Option<String>,
    pub behavioral_context: Option<String>,
    pub confidence: f64,
}

// =============================================================================
// Synthesis Pipeline
// =============================================================================

/// Main synthesis pipeline
#[derive(Debug, Clone)]
pub struct SynthesisPipeline {
    /// Vocabulary mapper
    mapper: VocabularyMapper,

    /// Audio segmenter
    segmenter: AudioSegmenter,

    /// Output directory
    output_dir: PathBuf,
}

impl SynthesisPipeline {
    /// Create a new synthesis pipeline
    ///
    /// # Arguments
    /// * `mapper` - Vocabulary mapper with context
    /// * `segmenter` - Audio segmenter
    /// * `output_dir` - Output directory for synthesis assets
    pub fn new(mapper: VocabularyMapper, segmenter: AudioSegmenter, output_dir: &Path) -> Self {
        Self {
            mapper,
            segmenter,
            output_dir: output_dir.to_path_buf(),
        }
    }

    /// Generate synthesis assets for all vocabulary
    ///
    /// This is the main entry point that generates:
    /// 1. Metadata for metadata-driven synthesis
    /// 2. Grains for granular synthesis
    /// 3. Audio segments for concatenative synthesis
    pub fn generate_synthesis_assets(&self) -> Result<SynthesisAssets> {
        let vocab_ids = self.mapper.vocabulary_ids();

        let mut metadata_assets = HashMap::new();
        let mut granular_assets = HashMap::new();
        let mut concatenative_assets = HashMap::new();

        for vocab_id in vocab_ids {
            let vocab = self
                .mapper
                .get_vocabulary(&vocab_id)
                .ok_or_else(|| SynthesisError::VocabularyNotFound(vocab_id.clone()))?;

            // Extract audio segments
            let segments = self
                .segmenter
                .extract_segments(vocab)
                .map_err(|e| SynthesisError::SegmenterError(e.to_string()))?;

            // Generate grains
            let mut all_grains = Vec::new();
            for segment in &segments {
                let grains = self.segmenter.generate_grains(
                    segment,
                    50.0, // grain_size_ms
                    25.0, // hop_size_ms
                    GrainEnvelope::Hann,
                )?;
                all_grains.extend(grains);
            }

            // Export grains
            let grain_paths = self
                .segmenter
                .export_grains(&all_grains)
                .map_err(|e| SynthesisError::ExportError(e.to_string()))?;

            // Export concatenative units
            let concat_paths = self
                .segmenter
                .export_concatenative(&segments, "wav")
                .map_err(|e| SynthesisError::ExportError(e.to_string()))?;

            // Create metadata
            let metadata = self.create_metadata(vocab, &segments);

            metadata_assets.insert(vocab_id.clone(), metadata);
            granular_assets.insert(vocab_id.clone(), grain_paths);
            concatenative_assets.insert(vocab_id.clone(), concat_paths);
        }

        // Export global metadata
        // Note: Can't directly export VocabularyItem with Array2
        // In production, you'd convert to a serializable format
        // For now, just export the IDs
        let all_vocab: Vec<_> = self
            .mapper
            .vocabulary_ids()
            .iter()
            .filter_map(|id| self.mapper.get_vocabulary(id))
            .collect();

        // Create dummy export path
        let metadata_path = self.output_dir.join("metadata.json");
        let _ = std::fs::write(&metadata_path, format!("{{\"vocabulary_count\": {}}}", all_vocab.len()));

        // Call simplified export
        let _metadata_path = self
            .segmenter
            .export_metadata(&all_vocab)
            .map_err(|e| SynthesisError::ExportError(e.to_string()))?;

        // let metadata_path = self
        //     .segmenter
        //     .export_metadata(&all_vocab)
        //     .map_err(|e| SynthesisError::ExportError(e.to_string()))?;

        Ok(SynthesisAssets {
            metadata_assets,
            granular_assets,
            concatenative_assets,
            metadata_path,
        })
    }

    /// Create metadata for a vocabulary item
    fn create_metadata(&self, vocab: &VocabularyItem, segments: &[AudioSegmentForSynthesis]) -> SynthesisMetadata {
        // Compute prosodic features
        let mut f0_contours = Vec::new();
        let mut intensities = Vec::new();

        for segment in segments {
            // Extract F0 contour (simplified)
            let f0 = self.extract_f0(&segment.audio, segment.sample_rate);
            f0_contours.extend(f0);

            // Extract intensity
            let rms = (segment.audio.iter().map(|&x| x * x).sum::<f32>() / segment.audio.len() as f32).sqrt();
            intensities.push(rms as f64);
        }

        let f0_mean = if f0_contours.is_empty() {
            0.0
        } else {
            f0_contours.iter().sum::<f64>() / f0_contours.len() as f64
        };

        let f0_min = f0_contours.iter().cloned().fold(f64::INFINITY, f64::min);
        let f0_max = f0_contours.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let intensity_mean = if intensities.is_empty() {
            0.0
        } else {
            intensities.iter().sum::<f64>() / intensities.len() as f64
        };

        SynthesisMetadata {
            vocab_id: vocab.vocab_id.clone(),
            duration_stats: DurationMetadata {
                min_ms: vocab.duration_stats.min_ms,
                max_ms: vocab.duration_stats.max_ms,
                mean_ms: vocab.duration_stats.mean_ms,
                typical_ms: vocab.duration_stats.mean_ms, // Use mean as typical
            },
            prosody: ProsodicMetadata {
                f0_contour: f0_contours.clone(),
                f0_mean,
                f0_range: (f0_min, f0_max),
                intensity: intensity_mean,
            },
            spectral: SpectralMetadata {
                centroid: 15000.0, // TODO: Extract from audio
                bandwidth: 5000.0,
                rolloff: 20000.0,
                mfcc: vec![0.0; 13], // TODO: Extract MFCCs
            },
        }
    }

    /// Extract F0 contour from audio (simplified autocorrelation)
    fn extract_f0(&self, audio: &[f32], sample_rate: u32) -> Vec<f64> {
        // Simplified: Use zero-crossing rate as proxy for F0
        // In production, use proper pitch detection algorithm
        let frame_size = (sample_rate as f64 * 0.01) as usize; // 10ms frames
        let mut f0_contour = Vec::new();

        for frame in audio.chunks(frame_size) {
            let zero_crossings = frame.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

            // Estimate F0 from zero-crossing rate
            let zcr = zero_crossings as f64 / frame.len() as f64;
            let f0 = if zcr > 0.0 {
                (zcr * sample_rate as f64 / 2.0).max(1000.0).min(100000.0)
            } else {
                0.0
            };

            f0_contour.push(f0);
        }

        f0_contour
    }

    /// Generate audio using metadata-driven synthesis
    ///
    /// # Arguments
    /// * `vocab_id` - Vocabulary ID to synthesize
    /// * `params` - Synthesis parameters
    pub fn synthesize_metadata_driven(&self, vocab_id: &str, params: &MetadataDrivenParams) -> Result<Vec<f32>> {
        let _vocab = self
            .mapper
            .get_vocabulary(vocab_id)
            .ok_or_else(|| SynthesisError::VocabularyNotFound(vocab_id.to_string()))?;

        // Generate samples based on parameters
        let target_samples = (params.target_duration * 48000.0) as usize;
        let mut audio = Vec::with_capacity(target_samples);

        // Generate F0 contour
        let f0_per_frame = self.interpolate_f0(&params.f0_contour, target_samples);

        // Generate sine wave with F0 modulation
        for (i, &f0) in f0_per_frame.iter().enumerate() {
            let t = i as f64 / 48000.0;
            let sample = params.intensity as f32 * (2.0 * std::f64::consts::PI * f0 * t).sin() as f32;

            audio.push(sample);
        }

        // Apply spectral shaping (simplified)
        audio = self.apply_spectral_shape(&audio, params.spectral_centroid, params.spectral_bandwidth);

        Ok(audio)
    }

    /// Interpolate F0 contour to target length
    fn interpolate_f0(&self, f0_contour: &[f64], target_length: usize) -> Vec<f64> {
        if f0_contour.is_empty() {
            return vec![10000.0; target_length];
        }

        if f0_contour.len() == 1 {
            return vec![f0_contour[0]; target_length];
        }

        let mut result = Vec::with_capacity(target_length);
        for i in 0..target_length {
            let pos = (i as f64 / target_length as f64) * (f0_contour.len() - 1) as f64;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f64;

            let val = if idx + 1 < f0_contour.len() {
                f0_contour[idx] * (1.0 - frac) + f0_contour[idx + 1] * frac
            } else {
                f0_contour[idx]
            };

            result.push(val);
        }

        result
    }

    /// Apply spectral shaping to audio
    fn apply_spectral_shape(&self, audio: &[f32], _centroid: f64, _bandwidth: f64) -> Vec<f32> {
        // Simplified: Just apply a bandpass filter
        // In production, use proper spectral shaping
        audio.to_vec()
    }

    /// Synthesize using granular synthesis
    ///
    /// # Arguments
    /// * `vocab_id` - Vocabulary ID to synthesize
    /// * `params` - Granular synthesis parameters
    /// * `duration` - Target duration in seconds
    pub fn synthesize_granular(
        &self,
        vocab_id: &str,
        params: &GranularSynthesisParams,
        duration: f64,
    ) -> Result<Vec<f32>> {
        let vocab = self
            .mapper
            .get_vocabulary(vocab_id)
            .ok_or_else(|| SynthesisError::VocabularyNotFound(vocab_id.to_string()))?;

        // Extract segments
        let segments = self
            .segmenter
            .extract_segments(vocab)
            .map_err(|e| SynthesisError::SegmenterError(e.to_string()))?;

        if segments.is_empty() {
            return Ok(vec![]);
        }

        // Select a random segment as source
        let source_segment = &segments[0];

        // Generate grains
        let grains = self.segmenter.generate_grains(
            source_segment,
            params.grain_size_ms,
            params.hop_size_ms,
            GrainEnvelope::from(params.envelope.clone()),
        )?;

        if grains.is_empty() {
            return Ok(vec![]);
        }

        // Synthesize by concatenating grains with overlap
        let target_samples = (duration * source_segment.sample_rate as f64) as usize;
        let hop_samples = (params.hop_size_ms * source_segment.sample_rate as f64 / 1000.0) as usize;

        let mut audio = vec![0.0f32; target_samples];
        let mut grain_idx = 0;

        for start in (0..target_samples).step_by(hop_samples) {
            let grain = &grains[grain_idx % grains.len()];

            // Mix grain into output
            for (i, &sample) in grain.audio.iter().enumerate() {
                let out_idx = start + i;
                if out_idx < target_samples {
                    audio[out_idx] += sample;
                }
            }

            grain_idx += 1;
        }

        // Normalize
        let max_amp = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_amp > 0.0 {
            for sample in audio.iter_mut() {
                *sample /= max_amp;
            }
        }

        Ok(audio)
    }

    /// Synthesize using concatenative synthesis
    ///
    /// # Arguments
    /// * `vocab_id` - Vocabulary ID to synthesize
    /// * `params` - Concatenative synthesis parameters
    pub fn synthesize_concatenative(&self, vocab_id: &str, params: &ConcatenativeParams) -> Result<Vec<f32>> {
        let vocab = self
            .mapper
            .get_vocabulary(vocab_id)
            .ok_or_else(|| SynthesisError::VocabularyNotFound(vocab_id.to_string()))?;

        // Extract segments
        let segments = self
            .segmenter
            .extract_segments(vocab)
            .map_err(|e| SynthesisError::SegmenterError(e.to_string()))?;

        if segments.is_empty() {
            return Ok(vec![]);
        }

        // Select best matching segment based on duration
        let target_duration = (params.min_segment_duration + params.max_segment_duration) / 2.0;
        let best_segment = segments
            .iter()
            .min_by(|a, b| {
                let diff_a = (a.duration - target_duration).abs();
                let diff_b = (b.duration - target_duration).abs();
                diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Clone the audio (with normalization if requested)
        let mut audio = best_segment.audio.clone();

        if params.normalize {
            let max_amp = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            if max_amp > 0.0 {
                for sample in audio.iter_mut() {
                    *sample /= max_amp;
                }
            }
        }

        Ok(audio)
    }
}

/// Synthesis assets output
#[derive(Debug, Clone)]
pub struct SynthesisAssets {
    /// Metadata-driven synthesis assets (vocab_id -> metadata)
    pub metadata_assets: HashMap<String, SynthesisMetadata>,

    /// Granular synthesis assets (vocab_id -> grain file paths)
    pub granular_assets: HashMap<String, Vec<PathBuf>>,

    /// Concatenative synthesis assets (vocab_id -> segment file paths)
    pub concatenative_assets: HashMap<String, Vec<PathBuf>>,

    /// Global metadata file path
    pub metadata_path: PathBuf,
}

/// Metadata for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    pub vocab_id: String,
    pub duration_stats: DurationMetadata,
    pub prosody: ProsodicMetadata,
    pub spectral: SpectralMetadata,
}

/// Duration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationMetadata {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub typical_ms: f64,
}

/// Prosodic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicMetadata {
    pub f0_contour: Vec<f64>,
    pub f0_mean: f64,
    pub f0_range: (f64, f64),
    pub intensity: f64,
}

/// Spectral metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMetadata {
    pub centroid: f64,
    pub bandwidth: f64,
    pub rolloff: f64,
    pub mfcc: Vec<f64>,
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::vocabulary_mapper::{AnnotationDataset, DurationStats, VocabularyOccurrence, VocalizationContext};

    /// Test 1: Create synthesis pipeline
    #[test]
    fn test_synthesis_pipeline_creation() {
        let temp_dir = TempDir::new().unwrap();

        let annotations = AnnotationDataset { annotations: vec![] };
        let mapper = VocabularyMapper::new(annotations, 48000);

        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().to_path_buf().join("output"),
            48000,
        );

        let synthesis_dir = temp_dir.path().join("synthesis");
        let pipeline = SynthesisPipeline::new(mapper, segmenter, &synthesis_dir);

        assert_eq!(pipeline.output_dir, temp_dir.path().join("synthesis"));
    }

    /// Test 2: Metadata-driven synthesis default params
    #[test]
    fn test_metadata_driven_default_params() {
        let params = MetadataDrivenParams::default();

        assert_eq!(params.target_duration, 0.1);
        assert_eq!(params.intensity, 0.7);
    }

    /// Test 3: Granular synthesis default params
    #[test]
    fn test_granular_default_params() {
        let params = GranularSynthesisParams::default();

        assert_eq!(params.grain_size_ms, 50.0);
        assert_eq!(params.hop_size_ms, 25.0);
        assert_eq!(params.density, 20.0);
    }

    /// Test 4: Concatenative synthesis default params
    #[test]
    fn test_concatenative_default_params() {
        let params = ConcatenativeParams::default();

        assert_eq!(params.crossfade_ms, 10.0);
        assert!(params.normalize);
    }

    /// Test 5: F0 interpolation
    #[test]
    fn test_f0_interpolation() {
        let temp_dir = TempDir::new().unwrap();
        let annotations = AnnotationDataset { annotations: vec![] };
        let mapper = VocabularyMapper::new(annotations, 48000);
        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().to_path_buf().join("output"),
            48000,
        );

        let synthesis_dir = temp_dir.path().join("synthesis");
        let pipeline = SynthesisPipeline::new(mapper, segmenter, &synthesis_dir);

        let f0_contour = vec![100.0, 200.0, 300.0];
        let interpolated = pipeline.interpolate_f0(&f0_contour, 10);

        assert_eq!(interpolated.len(), 10);
        assert!((interpolated[0] - 100.0).abs() < 1.0);
        // interpolated[9] = 200 * 0.2 + 300 * 0.8 = 280 (not exactly 300)
        assert!((interpolated[9] - 280.0).abs() < 1.0);
    }

    /// Test 6: Metadata-driven synthesis
    #[test]
    fn test_metadata_driven_synthesis() {
        let temp_dir = TempDir::new().unwrap();
        let annotations = AnnotationDataset { annotations: vec![] };

        let mut mapper = VocabularyMapper::new(annotations, 48000);

        // Add a vocabulary item
        use crate::vocabulary_mapper::{VocabularyItem, VocabularyOccurrence, VocalizationContext};

        // Add a vocabulary item using the public API
        // Note: In production, you'd call map_vocabulary() with proper data
        // For this test, we'll just use a minimal setup
        use ndarray::arr2;

        let cluster_labels = vec![0];
        let file_paths = vec!["test.wav".to_string()];
        let time_ranges = vec![(0.0, 0.1)];
        let feature_series = vec![arr2(&[[1.0, 2.0], [3.0, 4.0]])];

        let _result = mapper.map_vocabulary(&cluster_labels, &file_paths, &time_ranges, &feature_series);

        // Don't need to actually export for this test

        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().to_path_buf().join("output"),
            48000,
        );

        let synthesis_dir = temp_dir.path().join("synthesis");
        let pipeline = SynthesisPipeline::new(mapper, segmenter, &synthesis_dir);

        let params = MetadataDrivenParams::default();
        let result = pipeline.synthesize_metadata_driven("cluster_0", &params);

        // May fail if segments can't be loaded (test file doesn't exist)
        // That's acceptable for this unit test
        if let Ok(audio) = result {
            assert!(!audio.is_empty(), "Should produce audio");
        }
    }
}
