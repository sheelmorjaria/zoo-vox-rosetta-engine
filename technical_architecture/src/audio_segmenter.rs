// Audio Segmenter for Synthesis
//
// Extracts audio segments based on vocabulary mappings for use in:
// 1. Metadata-driven synthesis (parameter-based)
// 2. Granular synthesis (grain-based)
// 3. Concatenative synthesis (sample-based)
//
// This is the bridge between vocabulary discovery and synthesis execution.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// Re-use vocabulary types
use crate::vocabulary_mapper::{VocabularyItem, VocabularyOccurrence};

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SegmenterError {
    #[error("Audio file not found: {0}")]
    AudioNotFound(String),

    #[error("Failed to load audio: {0}")]
    AudioLoadError(String),

    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    #[error("Export failed: {0}")]
    ExportError(String),
}

pub type Result<T> = std::result::Result<T, SegmenterError>;

// =============================================================================
// Audio Segments
// =============================================================================

/// An extracted audio segment with metadata
#[derive(Debug, Clone)]
pub struct AudioSegmentForSynthesis {
    /// Unique segment ID
    pub segment_id: String,

    /// Vocabulary ID this segment belongs to
    pub vocab_id: String,

    /// Audio samples (mono, normalized to [-1, 1])
    pub audio: Vec<f32>,

    /// Sample rate
    pub sample_rate: u32,

    /// Duration in seconds
    pub duration: f64,

    /// Start time in original file
    pub start_time: f64,

    /// Source file path
    pub source_file: PathBuf,

    /// Contextual information
    pub context: SegmentContext,
}

/// Context information for a segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentContext {
    /// Which individual produced it
    pub emitter_id: Option<String>,

    /// Behavioral context
    pub behavioral_context: Option<String>,

    /// Time of day
    pub time_of_day: Option<String>,

    /// Confidence score
    pub confidence: f64,
}

/// Granular grain for granular synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGrain {
    /// Grain ID
    pub grain_id: String,

    /// Vocabulary ID
    pub vocab_id: String,

    /// Audio samples (typically short, 10-100ms)
    pub audio: Vec<f32>,

    /// Sample rate
    pub sample_rate: u32,

    /// Grain duration in seconds
    pub duration: f64,

    /// Grain envelope (for smooth concatenation)
    pub envelope: GrainEnvelope,

    /// Position in original segment
    pub position: f64,
}

/// Envelope type for grains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrainEnvelope {
    /// No envelope (raw audio)
    None,

    /// Linear fade in/out
    Linear { fade_in_ms: f64, fade_out_ms: f64 },

    /// Gaussian bell curve
    Gaussian { width_ms: f64 },

    /// Raised cosine (Hann window)
    Hann,
}

// =============================================================================
// Synthesis Metadata
// =============================================================================

/// Metadata for metadata-driven synthesis
#[derive(Debug, Clone)]
pub struct SynthesisMetadata {
    /// Vocabulary ID
    pub vocab_id: String,

    /// Acoustic features (30D) - flattened as Vec since Array2 isn't serializable
    pub features: Vec<Vec<f64>>,

    /// Duration statistics
    pub duration_stats: DurationMetadata,

    /// Prosodic features
    pub prosody: ProsodicMetadata,

    /// Spectral characteristics
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
    /// F0 contour (Hz)
    pub f0_contour: Vec<f64>,

    /// Average F0 (Hz)
    pub f0_mean: f64,

    /// F0 range (Hz)
    pub f0_range: (f64, f64),

    /// Intensity (RMS)
    pub intensity: f64,
}

/// Spectral metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMetadata {
    /// Centroid (Hz)
    pub centroid: f64,

    /// Bandwidth (Hz)
    pub bandwidth: f64,

    /// Spectral roll-off (Hz)
    pub rolloff: f64,

    /// MFCC coefficients
    pub mfcc: Vec<f64>,
}

// =============================================================================
// Audio Segmenter
// =============================================================================

/// Extracts audio segments for synthesis
#[derive(Debug, Clone)]
pub struct AudioSegmenter {
    /// Base audio directory
    audio_dir: PathBuf,

    /// Output directory for segments
    output_dir: PathBuf,

    /// Sample rate for processing
    sample_rate: u32,
}

impl AudioSegmenter {
    /// Create a new audio segmenter
    ///
    /// # Arguments
    /// * `audio_dir` - Base directory containing audio files
    /// * `output_dir` - Directory to write extracted segments
    /// * `sample_rate` - Target sample rate
    pub fn new<P: AsRef<Path>>(audio_dir: P, output_dir: P, sample_rate: u32) -> Self {
        Self {
            audio_dir: audio_dir.as_ref().to_path_buf(),
            output_dir: output_dir.as_ref().to_path_buf(),
            sample_rate,
        }
    }

    /// Extract audio segments from vocabulary mapping
    ///
    /// # Arguments
    /// * `vocabulary` - Vocabulary item with occurrences
    ///
    /// # Returns
    /// Vector of extracted audio segments
    pub fn extract_segments(
        &self,
        vocabulary: &VocabularyItem,
    ) -> Result<Vec<AudioSegmentForSynthesis>> {
        let mut segments = Vec::new();

        for (idx, occurrence) in vocabulary.occurrences.iter().enumerate() {
            let segment = self.extract_single_segment(vocabulary, occurrence, idx)?;
            segments.push(segment);
        }

        Ok(segments)
    }

    /// Extract a single audio segment
    fn extract_single_segment(
        &self,
        vocabulary: &VocabularyItem,
        occurrence: &VocabularyOccurrence,
        index: usize,
    ) -> Result<AudioSegmentForSynthesis> {
        let audio_path = self.audio_dir.join(&occurrence.file_path);

        if !audio_path.exists() {
            return Err(SegmenterError::AudioNotFound(
                audio_path.to_string_lossy().to_string(),
            ));
        }

        // Load audio using Symphonia
        let (full_audio, sr) = self.load_audio(&audio_path)?;

        // Extract segment
        let start = occurrence.start_sample;
        let end = occurrence.end_sample.min(full_audio.len());

        if start >= end {
            return Err(SegmenterError::InvalidAudioFormat(
                "Invalid sample range".to_string(),
            ));
        }

        let audio = full_audio[start..end].to_vec();

        let segment_id = format!("{}_seg_{}", vocabulary.vocab_id, index);

        Ok(AudioSegmentForSynthesis {
            segment_id,
            vocab_id: vocabulary.vocab_id.clone(),
            audio,
            sample_rate: sr,
            duration: (end - start) as f64 / sr as f64,
            start_time: occurrence.start_time,
            source_file: audio_path,
            context: SegmentContext {
                emitter_id: occurrence.context.emitter_id.clone(),
                behavioral_context: occurrence.context.behavioral_context.clone(),
                time_of_day: occurrence.context.time_of_day.clone(),
                confidence: occurrence.confidence,
            },
        })
    }

    /// Load audio file (simplified for TDD - just returns dummy data)
    fn load_audio(&self, path: &Path) -> Result<(Vec<f32>, u32)> {
        // For testing purposes, just return silence
        // In production, use hound or symphonia
        if !path.exists() {
            return Err(SegmenterError::AudioLoadError(format!(
                "Audio file not found: {:?}",
                path
            )));
        }

        // Return dummy audio (1 second of silence at 48kHz)
        Ok((vec![0.0f32; 48000], 48000))
    }

    /// Generate grains for granular synthesis
    ///
    /// # Arguments
    /// * `segment` - Audio segment to granulate
    /// * `grain_size_ms` - Size of each grain
    /// * `hop_size_ms` - Hop between grains
    /// * `envelope` - Envelope type
    pub fn generate_grains(
        &self,
        segment: &AudioSegmentForSynthesis,
        grain_size_ms: f64,
        hop_size_ms: f64,
        envelope: GrainEnvelope,
    ) -> Result<Vec<AudioGrain>> {
        let grain_size_samples = (grain_size_ms * segment.sample_rate as f64 / 1000.0) as usize;
        let hop_samples = (hop_size_ms * segment.sample_rate as f64 / 1000.0) as usize;

        let mut grains = Vec::new();

        for (idx, start) in (0..segment.audio.len()).step_by(hop_samples).enumerate() {
            let end = (start + grain_size_samples).min(segment.audio.len());

            if end - start < grain_size_samples / 2 {
                break; // Skip short final grain
            }

            let mut grain_audio = segment.audio[start..end].to_vec();

            // Apply envelope
            grain_audio = self.apply_envelope(&grain_audio, &envelope);

            let grain_id = format!("{}_grain_{}", segment.segment_id, idx);
            let position = start as f64 / segment.sample_rate as f64;

            grains.push(AudioGrain {
                grain_id,
                vocab_id: segment.vocab_id.clone(),
                audio: grain_audio,
                sample_rate: segment.sample_rate,
                duration: (end - start) as f64 / segment.sample_rate as f64,
                envelope: envelope.clone(),
                position,
            });
        }

        Ok(grains)
    }

    /// Apply envelope to audio
    fn apply_envelope(&self, audio: &[f32], envelope: &GrainEnvelope) -> Vec<f32> {
        match envelope {
            GrainEnvelope::None => audio.to_vec(),
            GrainEnvelope::Linear {
                fade_in_ms,
                fade_out_ms,
            } => {
                let fade_in_samples = (fade_in_ms * 1000.0) as usize;
                let fade_out_samples = (fade_out_ms * 1000.0) as usize;
                let len = audio.len();

                audio
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        if i < fade_in_samples {
                            x * (i as f64 / fade_in_samples as f64) as f32
                        } else if i > len - fade_out_samples {
                            let remaining = len - i;
                            x * (remaining as f64 / fade_out_samples as f64) as f32
                        } else {
                            x
                        }
                    })
                    .collect()
            }
            GrainEnvelope::Gaussian { width_ms } => {
                // Simplified Gaussian envelope
                let center = audio.len() / 2;
                let width = (width_ms * 1000.0) as usize;
                audio
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let dist = (i as f64 - center as f64) / width as f64;
                        x * (-dist * dist).exp() as f32
                    })
                    .collect()
            }
            GrainEnvelope::Hann => {
                // Hann window
                let len = audio.len();
                audio
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        x * (0.5
                            * (1.0
                                - (2.0 * std::f64::consts::PI * i as f64 / (len - 1) as f64).cos()))
                            as f32
                    })
                    .collect()
            }
        }
    }

    /// Export segments for concatenative synthesis
    ///
    /// # Arguments
    /// * `segments` - Segments to export
    /// * `format` - Export format (wav, flac, etc.)
    pub fn export_concatenative(
        &self,
        segments: &[AudioSegmentForSynthesis],
        format: &str,
    ) -> Result<Vec<PathBuf>> {
        let output_dir = self.output_dir.join("concatenative");
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| SegmenterError::ExportError(e.to_string()))?;

        let mut exported_paths = Vec::new();

        for segment in segments {
            let file_name = format!("{}.{}", segment.segment_id, format);
            let file_path = output_dir.join(&file_name);

            self.export_wav(&segment.audio, segment.sample_rate, &file_path)?;
            exported_paths.push(file_path);
        }

        Ok(exported_paths)
    }

    /// Export grains for granular synthesis
    ///
    /// # Arguments
    /// * `grains` - Grains to export
    pub fn export_grains(&self, grains: &[AudioGrain]) -> Result<Vec<PathBuf>> {
        let output_dir = self.output_dir.join("granular");
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| SegmenterError::ExportError(e.to_string()))?;

        let mut exported_paths = Vec::new();

        for grain in grains {
            let file_name = format!("{}.wav", grain.grain_id);
            let file_path = output_dir.join(&file_name);

            self.export_wav(&grain.audio, grain.sample_rate, &file_path)?;
            exported_paths.push(file_path);
        }

        Ok(exported_paths)
    }

    /// Export metadata for metadata-driven synthesis
    ///
    /// # Arguments
    /// * `vocabulary` - Vocabulary items with features
    pub fn export_metadata(&self, _vocabulary: &[&VocabularyItem]) -> Result<PathBuf> {
        let output_dir = self.output_dir.join("metadata");
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| SegmenterError::ExportError(e.to_string()))?;

        // Export as JSON (simplified - just vocab IDs and stats)
        // In production, you'd serialize VocabularyItem properly
        let json = r#"{"status": "metadata_export_simplified"}"#;

        let file_path = output_dir.join("synthesis_metadata.json");
        std::fs::write(&file_path, json).map_err(|e| SegmenterError::ExportError(e.to_string()))?;

        Ok(file_path)
    }

    /// Export audio as WAV file
    fn export_wav(&self, audio: &[f32], sample_rate: u32, path: &Path) -> Result<()> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let writer = WavWriter::create(path, spec)
            .map_err(|e| SegmenterError::ExportError(e.to_string()))?;

        let mut writer = writer;

        for &sample in audio {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| SegmenterError::ExportError(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| SegmenterError::ExportError(e.to_string()))?;

        Ok(())
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Test 1: Create audio segmenter
    #[test]
    fn test_audio_segmenter_creation() {
        let temp_dir = TempDir::new().unwrap();
        let audio_dir = temp_dir.path().to_path_buf();
        let output_dir = temp_dir.path().join("output");

        let segmenter = AudioSegmenter::new(&audio_dir, &output_dir, 48000);

        assert_eq!(segmenter.sample_rate, 48000);
    }

    /// Test 2: Grain generation
    #[test]
    fn test_generate_grains() {
        let temp_dir = TempDir::new().unwrap();
        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().to_path_buf().join("output"),
            48000,
        );

        let segment = AudioSegmentForSynthesis {
            segment_id: "test".to_string(),
            vocab_id: "cluster_0".to_string(),
            audio: vec![0.0f32; 4800], // 100ms at 48kHz
            sample_rate: 48000,
            duration: 0.1,
            start_time: 0.0,
            source_file: temp_dir.path().to_path_buf().join("test.wav"),
            context: SegmentContext {
                emitter_id: None,
                behavioral_context: None,
                time_of_day: None,
                confidence: 1.0,
            },
        };

        let grains = segmenter
            .generate_grains(&segment, 50.0, 25.0, GrainEnvelope::Hann)
            .unwrap();

        assert!(!grains.is_empty(), "Should generate grains");
    }

    /// Test 3: Envelope application
    #[test]
    fn test_apply_envelope() {
        let temp_dir = TempDir::new().unwrap();
        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().to_path_buf().join("output"),
            48000,
        );

        let audio = vec![1.0f32; 1000];
        let enveloped = segmenter.apply_envelope(&audio, &GrainEnvelope::None);

        assert_eq!(enveloped.len(), audio.len());
    }

    /// Test 4: Hann envelope shape
    #[test]
    fn test_hann_envelope() {
        let temp_dir = TempDir::new().unwrap();
        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().to_path_buf().join("output"),
            48000,
        );

        let audio = vec![1.0f32; 100];
        let enveloped = segmenter.apply_envelope(&audio, &GrainEnvelope::Hann);

        // Check that envelope tapers at edges
        assert!(enveloped[0] < 0.5, "Hann envelope should taper at start");
        assert!(enveloped[99] < 0.5, "Hann envelope should taper at end");
        assert!(enveloped[50] > 0.8, "Hann envelope should peak in middle");
    }

    /// Test 5: Grain envelope types
    #[test]
    fn test_grain_envelope_types() {
        // Just verify they compile and are distinct
        let env1 = GrainEnvelope::None;
        let env2 = GrainEnvelope::Linear {
            fade_in_ms: 5.0,
            fade_out_ms: 5.0,
        };
        let env3 = GrainEnvelope::Gaussian { width_ms: 10.0 };
        let env4 = GrainEnvelope::Hann;

        // All are valid
        assert!(matches!(env1, GrainEnvelope::None));
        assert!(matches!(env4, GrainEnvelope::Hann));
    }

    /// Test 6: Export metadata
    #[test]
    fn test_export_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().join("output"),
            48000,
        );

        use crate::vocabulary_mapper::{
            DurationStats, VocabularyItem, VocabularyOccurrence, VocalizationContext,
        };

        let vocab_item = VocabularyItem {
            vocab_id: "cluster_0".to_string(),
            cluster_id: 0,
            feature_templates: vec![],
            duration_stats: DurationStats {
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                std_ms: 0.0,
            },
            occurrences: vec![VocabularyOccurrence {
                file_path: "test.wav".to_string(),
                start_time: 0.0,
                end_time: 1.0,
                start_sample: 0,
                end_sample: 48000,
                context: VocalizationContext {
                    file_path: String::new(),
                    start_time: 0.0,
                    end_time: 1.0,
                    emitter_id: None,
                    addressee_id: None,
                    behavioral_context: None,
                    time_of_day: None,
                    location: None,
                    social_context: None,
                    environmental_conditions: None,
                },
                confidence: 1.0,
            }],
        };

        let vocab: Vec<&VocabularyItem> = vec![&vocab_item];
        let result = segmenter.export_metadata(&vocab);
        assert!(result.is_ok(), "Export should succeed");
    }

    /// Test 7: Empty audio handling
    #[test]
    fn test_empty_audio_handling() {
        let temp_dir = TempDir::new().unwrap();
        let segmenter = AudioSegmenter::new(
            temp_dir.path().to_path_buf(),
            temp_dir.path().join("output"),
            48000,
        );

        let segment = AudioSegmentForSynthesis {
            segment_id: "test".to_string(),
            vocab_id: "cluster_0".to_string(),
            audio: vec![],
            sample_rate: 48000,
            duration: 0.0,
            start_time: 0.0,
            source_file: temp_dir.path().join("test.wav"),
            context: SegmentContext {
                emitter_id: None,
                behavioral_context: None,
                time_of_day: None,
                confidence: 1.0,
            },
        };

        let grains = segmenter.generate_grains(&segment, 50.0, 25.0, GrainEnvelope::Hann);

        // Should handle empty audio gracefully
        assert!(grains.is_ok());
        assert!(grains.unwrap().is_empty());
    }
}
