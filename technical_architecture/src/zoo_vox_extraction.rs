//! Phrase Extraction Pipeline for Zoo Vox Rosetta Engine 2.0
//!
//! Implements phrase segmentation and extraction from audio,
//! with species-specific strategies based on encoding type.

use crate::species::SpeciesConfigFactory;
use crate::zoo_vox_data_models::{AcousticFeatures30D, BehaviorAnnotation, ContextAssociation, PhrasePrototype};
use crate::zoo_vox_features::ZooVoxFeatureExtractor;

/// Zoo Vox Rosetta extraction error type
#[derive(Debug)]
pub enum ZooVoxExtractionError {
    /// Species not found
    SpeciesNotFound(String),
    /// Audio processing error
    AudioProcessing(String),
    /// Feature extraction error
    FeatureExtraction(String),
}

impl std::fmt::Display for ZooVoxExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZooVoxExtractionError::SpeciesNotFound(s) => write!(f, "Species not found: {}", s),
            ZooVoxExtractionError::AudioProcessing(msg) => {
                write!(f, "Audio processing error: {}", msg)
            }
            ZooVoxExtractionError::FeatureExtraction(msg) => {
                write!(f, "Feature extraction error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ZooVoxExtractionError {}

/// Configuration for phrase extraction
#[derive(Debug, Clone)]
pub struct ZooVoxExtractionConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Minimum phrase duration in ms
    pub min_phrase_duration_ms: f64,
    /// Maximum phrase duration in ms
    pub max_phrase_duration_ms: f64,
    /// Silence threshold in dB
    pub silence_threshold_db: f64,
    /// Minimum silence gap in ms
    pub min_silence_gap_ms: f64,
    /// F0 bin size for phrase key generation
    pub f0_bin_size_hz: f64,
    /// Duration bin size for phrase key generation
    pub duration_bin_size_ms: f64,
    /// Similarity threshold for phrase typing
    pub similarity_threshold: f64,
    /// Minimum signal-to-noise ratio
    pub min_snr_db: f64,
    /// Minimum extraction confidence
    pub min_extraction_confidence: f64,
}

impl Default for ZooVoxExtractionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            min_phrase_duration_ms: 10.0,
            max_phrase_duration_ms: 2000.0,
            silence_threshold_db: -40.0,
            min_silence_gap_ms: 20.0,
            f0_bin_size_hz: 200.0,
            duration_bin_size_ms: 10.0,
            similarity_threshold: 0.85,
            min_snr_db: 10.0,
            min_extraction_confidence: 0.7,
        }
    }
}

impl ZooVoxExtractionConfig {
    /// Create new configuration with sample rate
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }

    /// Get species-specific configuration
    pub fn for_species(species: &str, sample_rate: u32) -> Self {
        let config = SpeciesConfigFactory::create(species);
        let params = config.feature_params();

        Self {
            sample_rate,
            min_phrase_duration_ms: params.phrase_min_ms,
            max_phrase_duration_ms: params.phrase_max_ms,
            silence_threshold_db: -40.0,
            min_silence_gap_ms: 20.0,
            f0_bin_size_hz: 200.0,
            duration_bin_size_ms: 10.0,
            similarity_threshold: params.similarity_threshold,
            min_snr_db: 10.0,
            min_extraction_confidence: 0.7,
        }
    }
}

/// Phrase extractor for segmenting and typing vocalizations
pub struct ZooVoxPhraseExtractor {
    config: ZooVoxExtractionConfig,
    feature_extractor: ZooVoxFeatureExtractor,
}

impl ZooVoxPhraseExtractor {
    /// Create new phrase extractor
    pub fn new(config: ZooVoxExtractionConfig) -> Self {
        let feature_extractor = ZooVoxFeatureExtractor::new(config.sample_rate);
        Self {
            config,
            feature_extractor,
        }
    }

    /// Create with default configuration
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self::new(ZooVoxExtractionConfig::new(sample_rate))
    }

    /// Create for specific species
    pub fn for_species(species: &str, sample_rate: u32) -> Self {
        Self::new(ZooVoxExtractionConfig::for_species(species, sample_rate))
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Extract phrases from audio for a specific species
    pub fn extract_phrases(
        &mut self,
        audio: &[f64],
        species: &str,
        annotations: Option<&[BehaviorAnnotation]>,
    ) -> Result<Vec<PhrasePrototype>, ZooVoxExtractionError> {
        let config = SpeciesConfigFactory::create(species);

        // Segment audio based on encoding strategy
        let segments = self.segment_audio(audio, &config.encoding_strategy().to_string())?;

        // Extract features for each segment
        let mut phrases = Vec::new();

        for (i, (start, end)) in segments.into_iter().enumerate() {
            if end <= start {
                continue;
            }

            let segment_audio = &audio[start..end];

            // Skip very short segments
            let duration_ms = (segment_audio.len() as f64 / self.config.sample_rate as f64) * 1000.0;
            if duration_ms < self.config.min_phrase_duration_ms {
                continue;
            }

            // Extract 30D features
            let features = match self.feature_extractor.extract(segment_audio) {
                Ok(f) => f,
                Err(_) => continue, // Skip problematic segments
            };

            // Generate phrase key
            let phrase_key = self.generate_phrase_key(&features);

            // Find associated context
            let context = annotations
                .as_ref()
                .and_then(|anns| self.find_associated_context(start, end, anns));

            // Create phrase prototype
            let mut phrase = PhrasePrototype::new(
                format!("{}_{}", species.replace(" ", "_").to_lowercase(), i),
                phrase_key,
                species.to_string(),
            );

            phrase.encoding_strategy = config.encoding_strategy();
            phrase.encoding_modality = config.modality();
            phrase.features_30d = features;
            phrase.primary_context = context.as_ref().map(|c| c.context_label.clone());

            if let Some(ctx) = context {
                phrase.contexts.push(ctx);
            }

            phrase.occurrence_count = 1;

            phrases.push(phrase);
        }

        Ok(phrases)
    }

    /// Segment audio based on species encoding strategy
    fn segment_audio(
        &self,
        audio: &[f64],
        encoding_strategy: &str,
    ) -> Result<Vec<(usize, usize)>, ZooVoxExtractionError> {
        match encoding_strategy.to_lowercase().as_str() {
            "coda-type" => self.segment_codas(audio),
            "frequency-modulated" => self.segment_whistles(audio),
            "combinatorial" => self.segment_combinatorial(audio),
            _ => self.segment_by_silence(audio),
        }
    }

    /// Segment sperm whale codas (click patterns)
    fn segment_codas(&self, audio: &[f64]) -> Result<Vec<(usize, usize)>, ZooVoxExtractionError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // High-frequency click detection
        let threshold = self.percentile(audio, 95.0);

        let mut segments = Vec::new();
        let mut in_segment = false;
        let mut segment_start = 0;

        for (i, &sample) in audio.iter().enumerate() {
            let above = sample.abs() > threshold;

            if above && !in_segment {
                in_segment = true;
                segment_start = i;
            } else if !above && in_segment {
                in_segment = false;
                segments.push((segment_start, i));
            }
        }

        // Don't forget last segment
        if in_segment {
            segments.push((segment_start, audio.len()));
        }

        Ok(segments)
    }

    /// Segment dolphin whistles (FM contours)
    fn segment_whistles(&self, audio: &[f64]) -> Result<Vec<(usize, usize)>, ZooVoxExtractionError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // For FM-modulated vocalizations, segment based on continuity
        let envelope: Vec<f64> = audio.iter().map(|x| x.abs()).collect();
        let threshold = self.percentile(&envelope, 30.0);

        let mut segments = Vec::new();
        let mut in_segment = false;
        let mut segment_start = 0;

        for (i, &val) in envelope.iter().enumerate() {
            if val > threshold && !in_segment {
                in_segment = true;
                segment_start = i;
            } else if val <= threshold && in_segment {
                in_segment = false;
                segments.push((segment_start, i));
            }
        }

        if in_segment {
            segments.push((segment_start, audio.len()));
        }

        Ok(segments)
    }

    /// Segment combinatorial vocalizations (bird songs, orca calls)
    fn segment_combinatorial(&self, audio: &[f64]) -> Result<Vec<(usize, usize)>, ZooVoxExtractionError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // Energy-based segmentation with adaptive threshold
        let envelope: Vec<f64> = audio.iter().map(|x| x.abs()).collect();

        let mean = envelope.iter().sum::<f64>() / envelope.len() as f64;
        let std = {
            let variance: f64 = envelope.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / envelope.len() as f64;
            variance.sqrt()
        };

        let threshold = mean + 0.5 * std;

        let min_gap = (self.config.sample_rate as f64 * self.config.min_silence_gap_ms / 1000.0) as usize;

        let mut segments = Vec::new();
        let mut in_segment = false;
        let mut segment_start = 0;
        let mut gap_count = 0;

        for (i, &val) in envelope.iter().enumerate() {
            if val > threshold {
                if !in_segment {
                    in_segment = true;
                    segment_start = i;
                }
                gap_count = 0;
            } else if in_segment {
                gap_count += 1;
                if gap_count > min_gap {
                    segments.push((segment_start, i - gap_count));
                    in_segment = false;
                    gap_count = 0;
                }
            }
        }

        if in_segment {
            segments.push((segment_start, audio.len()));
        }

        Ok(segments)
    }

    /// Default segmentation by silence gaps
    fn segment_by_silence(&self, audio: &[f64]) -> Result<Vec<(usize, usize)>, ZooVoxExtractionError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let envelope: Vec<f64> = audio.iter().map(|x| x.abs()).collect();

        let max_val = envelope.iter().fold(0.0_f64, |a, &b| a.max(b));
        let threshold_db = self.config.silence_threshold_db;
        let threshold = max_val * 10.0_f64.powf(threshold_db / 20.0);

        let min_gap = (self.config.sample_rate as f64 * self.config.min_silence_gap_ms / 1000.0) as usize;
        let min_duration = (self.config.sample_rate as f64 * self.config.min_phrase_duration_ms / 1000.0) as usize;

        let mut segments = Vec::new();
        let mut in_segment = false;
        let mut segment_start = 0;
        let mut gap_count = 0;

        for (i, &val) in envelope.iter().enumerate() {
            if val > threshold {
                if !in_segment {
                    in_segment = true;
                    segment_start = i;
                }
                gap_count = 0;
            } else if in_segment {
                gap_count += 1;
                if gap_count > min_gap {
                    let end = i - gap_count;
                    if end - segment_start >= min_duration {
                        segments.push((segment_start, end));
                    }
                    in_segment = false;
                    gap_count = 0;
                }
            }
        }

        if in_segment {
            let end = audio.len();
            if end - segment_start >= min_duration {
                segments.push((segment_start, end));
            }
        }

        Ok(segments)
    }

    /// Generate human-readable phrase key
    fn generate_phrase_key(&self, features: &AcousticFeatures30D) -> String {
        let f0_bin = (features.mean_f0_hz / self.config.f0_bin_size_hz).floor() * self.config.f0_bin_size_hz;
        let dur_bin =
            (features.duration_ms / self.config.duration_bin_size_ms).floor() * self.config.duration_bin_size_ms;
        format!("F0_{:.0}_DUR_{:.0}", f0_bin, dur_bin)
    }

    /// Find context annotation overlapping with segment
    fn find_associated_context(
        &self,
        start: usize,
        end: usize,
        annotations: &[BehaviorAnnotation],
    ) -> Option<ContextAssociation> {
        for ann in annotations {
            let ann_start = (ann.start_seconds * self.config.sample_rate as f64) as usize;
            let ann_end = (ann.end_seconds * self.config.sample_rate as f64) as usize;

            // Check overlap
            if start < ann_end && end > ann_start {
                return Some(ContextAssociation::new(&ann.context_label, &ann.context_category));
            }
        }
        None
    }

    /// Compute percentile of array
    fn percentile(&self, data: &[f64], percentile: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_config_default() {
        let config = ZooVoxExtractionConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.min_phrase_duration_ms, 10.0);
    }

    #[test]
    fn test_phrase_extractor_creation() {
        let config = ZooVoxExtractionConfig::new(44100);
        let extractor = ZooVoxPhraseExtractor::new(config);
        assert_eq!(extractor.config.sample_rate, 44100);
    }

    #[test]
    fn test_segment_silence_empty() {
        let config = ZooVoxExtractionConfig::default();
        let extractor = ZooVoxPhraseExtractor::new(config);

        let segments = extractor.segment_by_silence(&[]).unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn test_generate_phrase_key() {
        let config = ZooVoxExtractionConfig::default();
        let extractor = ZooVoxPhraseExtractor::new(config);

        let features = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let key = extractor.generate_phrase_key(&features);
        assert!(key.starts_with("F0_"));
        assert!(key.contains("_DUR_"));
    }

    #[test]
    fn test_extract_phrases_sine_wave() {
        let config = ZooVoxExtractionConfig::new(48000);
        let mut extractor = ZooVoxPhraseExtractor::new(config);

        // Generate 1 second of 1000 Hz sine wave
        let audio: Vec<f64> = (0..48000)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin() * 0.5)
            .collect();

        let phrases = extractor.extract_phrases(&audio, "marmoset", None).unwrap();

        // Should extract at least one phrase
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_behavior_annotation() {
        let ann = BehaviorAnnotation::new(0.0, 1.0, "contact", "social");
        assert_eq!(ann.context_label, "contact");
        assert_eq!(ann.context_category, "social");
    }

    #[test]
    fn test_config_for_species() {
        let config = ZooVoxExtractionConfig::for_species("sperm_whale", 48000);
        assert!(config.min_phrase_duration_ms < 50.0); // Sperm whales have short phrases
    }
}
