//! Data models for Zoo Vox Rosetta Engine 2.0 phrase data structures
//!
//! This module defines the core data structures used throughout the
//! Zoo Vox Rosetta Engine for representing phrases, features, and context.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// 45D ACOUSTIC FEATURES
// ============================================================================

/// 45-Dimensional Extended Feature Vector
///
/// Organized into 9 categories:
/// - Fundamental (3): Basic acoustic parameters
/// - Grit Factors (3): Timbre texture
/// - Motion Factors (7): Envelope dynamics
/// - Fingerprint Factors (14): Spectral shape
/// - Rhythm Factors (3): Temporal patterns
/// - Resonance Factors (6): Vocal tract geometry (NEW)
/// - Spectral Shape Factors (4): Energy distribution (NEW)
/// - Modulation Factors (3): Sweep/flutter analysis (NEW)
/// - Non-Linear Factors (2): Chaos detection (NEW)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AcousticFeatures45D {
    // === FUNDAMENTAL (3 features) ===
    /// Mean fundamental frequency (Hz)
    pub mean_f0_hz: f64,
    /// Phrase duration (ms)
    pub duration_ms: f64,
    /// F0 variation range (Hz)
    pub f0_range_hz: f64,

    // === GRIT FACTORS (3 features) - Timbre texture ===
    /// Harmonic-to-noise ratio (dB)
    pub harmonic_to_noise_ratio: f64,
    /// Wiener entropy (0=tonal, 1=noise)
    pub spectral_flatness: f64,
    /// Harmonic coherence (0-1)
    pub harmonicity: f64,

    // === MOTION FACTORS (7 features) - Envelope dynamics ===
    /// Attack phase duration (ms)
    pub attack_time_ms: f64,
    /// Decay phase duration (ms)
    pub decay_time_ms: f64,
    /// Sustain amplitude (0-1)
    pub sustain_level: f64,
    /// Vibrato frequency (Hz)
    pub vibrato_rate_hz: f64,
    /// Vibrato depth (semitones)
    pub vibrato_depth: f64,
    /// Frequency perturbation
    pub jitter: f64,
    /// Amplitude perturbation
    pub shimmer: f64,

    // === FINGERPRINT FACTORS (14 features) - Spectral shape ===
    /// MFCC coefficients 1-13
    pub mfcc_1: f64,
    pub mfcc_2: f64,
    pub mfcc_3: f64,
    pub mfcc_4: f64,
    pub mfcc_5: f64,
    pub mfcc_6: f64,
    pub mfcc_7: f64,
    pub mfcc_8: f64,
    pub mfcc_9: f64,
    pub mfcc_10: f64,
    pub mfcc_11: f64,
    pub mfcc_12: f64,
    pub mfcc_13: f64,
    /// Spectral change rate
    pub spectral_flux: f64,

    // === RHYTHM FACTORS (3 features) - Temporal patterns ===
    /// Inter-click interval median (ms)
    pub median_ici_ms: f64,
    /// Onset event rate (Hz)
    pub onset_rate_hz: f64,
    /// ICI variability
    pub ici_coefficient_of_variation: f64,

    // === RESONANCE FACTORS (6 features) - Vocal tract geometry (NEW) ===
    /// First formant frequency (Hz) - vocal tract openness
    pub formant_1_hz: f64,
    /// Second formant frequency (Hz) - tongue position
    pub formant_2_hz: f64,
    /// Third formant frequency (Hz) - complex oral shapes
    pub formant_3_hz: f64,
    /// First formant bandwidth (Hz) - breathiness indicator
    pub formant_1_bandwidth: f64,
    /// Second formant bandwidth (Hz) - damping indicator
    pub formant_2_bandwidth: f64,
    /// Average spacing between formants (Hz) - vocal tract length estimate
    pub formant_dispersion: f64,

    // === SPECTRAL SHAPE FACTORS (4 features) - Energy distribution (NEW) ===
    /// Spectral centroid (Hz) - brightness
    pub spectral_centroid: f64,
    /// Spectral spread (Hz) - bandwidth
    pub spectral_spread: f64,
    /// Spectral skewness - asymmetry of spectrum
    pub spectral_skewness: f64,
    /// Spectral kurtosis - peakedness/texture
    pub spectral_kurtosis: f64,

    // === MODULATION FACTORS (3 features) - Sweep/flutter analysis (NEW) ===
    /// Spectral tilt (dB/octave) - vocal effort
    pub spectral_tilt: f64,
    /// FM slope (Hz/sec) - frequency sweep rate (dolphin/bat)
    pub fm_slope_hz_per_sec: f64,
    /// AM depth (0-1) - tremolo intensity
    pub am_depth: f64,

    // === NON-LINEAR FACTORS (2 features) - Chaos detection (NEW) ===
    /// Subharmonic ratio - biphonation detection
    pub subharmonic_ratio: f64,
    /// Spectral entropy - chaos vs noise
    pub spectral_entropy: f64,
}

impl AcousticFeatures45D {
    /// Create new empty features
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to 45D vector
    pub fn to_vector(&self) -> [f64; 45] {
        [
            // Fundamental (3)
            self.mean_f0_hz,
            self.duration_ms,
            self.f0_range_hz,
            // Grit (3)
            self.harmonic_to_noise_ratio,
            self.spectral_flatness,
            self.harmonicity,
            // Motion (7)
            self.attack_time_ms,
            self.decay_time_ms,
            self.sustain_level,
            self.vibrato_rate_hz,
            self.vibrato_depth,
            self.jitter,
            self.shimmer,
            // Fingerprint (14)
            self.mfcc_1,
            self.mfcc_2,
            self.mfcc_3,
            self.mfcc_4,
            self.mfcc_5,
            self.mfcc_6,
            self.mfcc_7,
            self.mfcc_8,
            self.mfcc_9,
            self.mfcc_10,
            self.mfcc_11,
            self.mfcc_12,
            self.mfcc_13,
            self.spectral_flux,
            // Rhythm (3)
            self.median_ici_ms,
            self.onset_rate_hz,
            self.ici_coefficient_of_variation,
            // Resonance (6) - NEW
            self.formant_1_hz,
            self.formant_2_hz,
            self.formant_3_hz,
            self.formant_1_bandwidth,
            self.formant_2_bandwidth,
            self.formant_dispersion,
            // Spectral Shape (4) - NEW
            self.spectral_centroid,
            self.spectral_spread,
            self.spectral_skewness,
            self.spectral_kurtosis,
            // Modulation (3) - NEW
            self.spectral_tilt,
            self.fm_slope_hz_per_sec,
            self.am_depth,
            // Non-Linear (2) - NEW
            self.subharmonic_ratio,
            self.spectral_entropy,
        ]
    }

    /// Create from 45D vector
    pub fn from_vector(vec: [f64; 45]) -> Self {
        Self {
            // Fundamental (3)
            mean_f0_hz: vec[0],
            duration_ms: vec[1],
            f0_range_hz: vec[2],
            // Grit (3)
            harmonic_to_noise_ratio: vec[3],
            spectral_flatness: vec[4],
            harmonicity: vec[5],
            // Motion (7)
            attack_time_ms: vec[6],
            decay_time_ms: vec[7],
            sustain_level: vec[8],
            vibrato_rate_hz: vec[9],
            vibrato_depth: vec[10],
            jitter: vec[11],
            shimmer: vec[12],
            // Fingerprint (14)
            mfcc_1: vec[13],
            mfcc_2: vec[14],
            mfcc_3: vec[15],
            mfcc_4: vec[16],
            mfcc_5: vec[17],
            mfcc_6: vec[18],
            mfcc_7: vec[19],
            mfcc_8: vec[20],
            mfcc_9: vec[21],
            mfcc_10: vec[22],
            mfcc_11: vec[23],
            mfcc_12: vec[24],
            mfcc_13: vec[25],
            spectral_flux: vec[26],
            // Rhythm (3)
            median_ici_ms: vec[27],
            onset_rate_hz: vec[28],
            ici_coefficient_of_variation: vec[29],
            // Resonance (6) - NEW
            formant_1_hz: vec[30],
            formant_2_hz: vec[31],
            formant_3_hz: vec[32],
            formant_1_bandwidth: vec[33],
            formant_2_bandwidth: vec[34],
            formant_dispersion: vec[35],
            // Spectral Shape (4) - NEW
            spectral_centroid: vec[36],
            spectral_spread: vec[37],
            spectral_skewness: vec[38],
            spectral_kurtosis: vec[39],
            // Modulation (3) - NEW
            spectral_tilt: vec[40],
            fm_slope_hz_per_sec: vec[41],
            am_depth: vec[42],
            // Non-Linear (2) - NEW
            subharmonic_ratio: vec[43],
            spectral_entropy: vec[44],
        }
    }

    /// Compute Euclidean distance to another feature vector
    pub fn distance(&self, other: &AcousticFeatures45D) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute cosine similarity to another feature vector
    pub fn cosine_similarity(&self, other: &AcousticFeatures45D) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Convert from 30D features (fills new fields with defaults)
    pub fn from_30d(features_30d: &AcousticFeatures30D) -> Self {
        Self {
            // Copy all 30D fields
            mean_f0_hz: features_30d.mean_f0_hz,
            duration_ms: features_30d.duration_ms,
            f0_range_hz: features_30d.f0_range_hz,
            harmonic_to_noise_ratio: features_30d.harmonic_to_noise_ratio,
            spectral_flatness: features_30d.spectral_flatness,
            harmonicity: features_30d.harmonicity,
            attack_time_ms: features_30d.attack_time_ms,
            decay_time_ms: features_30d.decay_time_ms,
            sustain_level: features_30d.sustain_level,
            vibrato_rate_hz: features_30d.vibrato_rate_hz,
            vibrato_depth: features_30d.vibrato_depth,
            jitter: features_30d.jitter,
            shimmer: features_30d.shimmer,
            mfcc_1: features_30d.mfcc_1,
            mfcc_2: features_30d.mfcc_2,
            mfcc_3: features_30d.mfcc_3,
            mfcc_4: features_30d.mfcc_4,
            mfcc_5: features_30d.mfcc_5,
            mfcc_6: features_30d.mfcc_6,
            mfcc_7: features_30d.mfcc_7,
            mfcc_8: features_30d.mfcc_8,
            mfcc_9: features_30d.mfcc_9,
            mfcc_10: features_30d.mfcc_10,
            mfcc_11: features_30d.mfcc_11,
            mfcc_12: features_30d.mfcc_12,
            mfcc_13: features_30d.mfcc_13,
            spectral_flux: features_30d.spectral_flux,
            median_ici_ms: features_30d.median_ici_ms,
            onset_rate_hz: features_30d.onset_rate_hz,
            ici_coefficient_of_variation: features_30d.ici_coefficient_of_variation,
            // New 15D fields default to 0.0
            ..Default::default()
        }
    }
}

// ============================================================================
// 30D ACOUSTIC FEATURES
// ============================================================================

/// 30-Dimensional Micro-Dynamics Feature Vector
///
/// Organized into 5 categories:
/// - Fundamental (3): Basic acoustic parameters
/// - Grit Factors (3): Timbre texture
/// - Motion Factors (7): Envelope dynamics
/// - Fingerprint Factors (14): Spectral shape
/// - Rhythm Factors (3): Temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AcousticFeatures30D {
    // === FUNDAMENTAL (3 features) ===
    /// Mean fundamental frequency (Hz)
    pub mean_f0_hz: f64,
    /// Phrase duration (ms)
    pub duration_ms: f64,
    /// F0 variation range (Hz)
    pub f0_range_hz: f64,

    // === GRIT FACTORS (3 features) - Timbre texture ===
    /// Harmonic-to-noise ratio (dB)
    pub harmonic_to_noise_ratio: f64,
    /// Wiener entropy (0=tonal, 1=noise)
    pub spectral_flatness: f64,
    /// Harmonic coherence (0-1)
    pub harmonicity: f64,

    // === MOTION FACTORS (7 features) - Envelope dynamics ===
    /// Attack phase duration (ms)
    pub attack_time_ms: f64,
    /// Decay phase duration (ms)
    pub decay_time_ms: f64,
    /// Sustain amplitude (0-1)
    pub sustain_level: f64,
    /// Vibrato frequency (Hz)
    pub vibrato_rate_hz: f64,
    /// Vibrato depth (semitones)
    pub vibrato_depth: f64,
    /// Frequency perturbation
    pub jitter: f64,
    /// Amplitude perturbation
    pub shimmer: f64,

    // === FINGERPRINT FACTORS (14 features) - Spectral shape ===
    /// MFCC coefficients 1-13
    pub mfcc_1: f64,
    pub mfcc_2: f64,
    pub mfcc_3: f64,
    pub mfcc_4: f64,
    pub mfcc_5: f64,
    pub mfcc_6: f64,
    pub mfcc_7: f64,
    pub mfcc_8: f64,
    pub mfcc_9: f64,
    pub mfcc_10: f64,
    pub mfcc_11: f64,
    pub mfcc_12: f64,
    pub mfcc_13: f64,
    /// Spectral change rate
    pub spectral_flux: f64,

    // === RHYTHM FACTORS (3 features) - Temporal patterns ===
    /// Inter-click interval median (ms)
    pub median_ici_ms: f64,
    /// Onset event rate (Hz)
    pub onset_rate_hz: f64,
    /// ICI variability
    pub ici_coefficient_of_variation: f64,
}

impl AcousticFeatures30D {
    /// Create new empty features
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to 30D vector
    pub fn to_vector(&self) -> [f64; 30] {
        [
            // Fundamental
            self.mean_f0_hz,
            self.duration_ms,
            self.f0_range_hz,
            // Grit
            self.harmonic_to_noise_ratio,
            self.spectral_flatness,
            self.harmonicity,
            // Motion
            self.attack_time_ms,
            self.decay_time_ms,
            self.sustain_level,
            self.vibrato_rate_hz,
            self.vibrato_depth,
            self.jitter,
            self.shimmer,
            // Fingerprint
            self.mfcc_1,
            self.mfcc_2,
            self.mfcc_3,
            self.mfcc_4,
            self.mfcc_5,
            self.mfcc_6,
            self.mfcc_7,
            self.mfcc_8,
            self.mfcc_9,
            self.mfcc_10,
            self.mfcc_11,
            self.mfcc_12,
            self.mfcc_13,
            self.spectral_flux,
            // Rhythm
            self.median_ici_ms,
            self.onset_rate_hz,
            self.ici_coefficient_of_variation,
        ]
    }

    /// Create from 30D vector
    pub fn from_vector(vec: [f64; 30]) -> Self {
        Self {
            // Fundamental
            mean_f0_hz: vec[0],
            duration_ms: vec[1],
            f0_range_hz: vec[2],
            // Grit
            harmonic_to_noise_ratio: vec[3],
            spectral_flatness: vec[4],
            harmonicity: vec[5],
            // Motion
            attack_time_ms: vec[6],
            decay_time_ms: vec[7],
            sustain_level: vec[8],
            vibrato_rate_hz: vec[9],
            vibrato_depth: vec[10],
            jitter: vec[11],
            shimmer: vec[12],
            // Fingerprint
            mfcc_1: vec[13],
            mfcc_2: vec[14],
            mfcc_3: vec[15],
            mfcc_4: vec[16],
            mfcc_5: vec[17],
            mfcc_6: vec[18],
            mfcc_7: vec[19],
            mfcc_8: vec[20],
            mfcc_9: vec[21],
            mfcc_10: vec[22],
            mfcc_11: vec[23],
            mfcc_12: vec[24],
            mfcc_13: vec[25],
            spectral_flux: vec[26],
            // Rhythm
            median_ici_ms: vec[27],
            onset_rate_hz: vec[28],
            ici_coefficient_of_variation: vec[29],
        }
    }

    /// Compute Euclidean distance to another feature vector
    pub fn distance(&self, other: &AcousticFeatures30D) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute cosine similarity to another feature vector
    pub fn cosine_similarity(&self, other: &AcousticFeatures30D) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

// ============================================================================
// CONTEXT ASSOCIATION
// ============================================================================

/// Context-to-Phrase Association
///
/// Links phrases to behavioral contexts based on discovered encoding strategies.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextAssociation {
    /// Context label (e.g., "alarm", "feeding", "contact")
    pub context_label: String,
    /// Context category (e.g., "defensive", "foraging", "social")
    pub context_category: String,
    /// Number of times phrase appears in this context
    #[serde(default)]
    pub occurrence_count: u32,
    /// P(context | phrase)
    #[serde(default)]
    pub context_probability: f64,
    /// P(phrase | context)
    #[serde(default)]
    pub phrase_probability: f64,

    // === Encoding-specific fields ===
    /// For quantitative encoding: phrase count in context
    pub phrase_count_in_context: Option<u32>,
    /// For combinatorial encoding: position in sequence
    pub sequence_position: Option<u32>,
    /// For duration-mediated encoding: duration bin
    pub duration_bin: Option<String>,
    /// For FM-modulated encoding: contour type
    pub fm_contour_type: Option<String>,
}

impl ContextAssociation {
    /// Create new context association
    pub fn new(label: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            context_label: label.into(),
            context_category: category.into(),
            ..Default::default()
        }
    }
}

// ============================================================================
// PHRASE PROTOTYPE
// ============================================================================

/// Phrase Prototype for Zoo Vox Rosetta Engine 2.0
///
/// Represents a single phrase with full 30D features and context associations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhrasePrototype {
    // === Identification ===
    /// Unique identifier
    pub phrase_id: String,
    /// Human-readable key (e.g., "F0_6400_DUR_50")
    pub phrase_key: String,
    /// Species name
    pub species: String,
    /// Original audio file
    pub source_file: Option<String>,
    /// Dataset origin
    pub source_dataset: Option<String>,

    // === Classification ===
    /// Encoding strategy used (reuse from species module)
    #[serde(with = "encoding_strategy_serde")]
    pub encoding_strategy: crate::species::EncodingStrategy,
    /// Temporal or Spectral
    #[serde(with = "encoding_modality_serde")]
    pub encoding_modality: crate::species::AnalysisModality,
    /// Discrete type label
    pub phrase_type: Option<String>,

    // === Features ===
    /// 30D acoustic features
    pub features_30d: AcousticFeatures30D,

    // === Context Associations ===
    /// Context associations
    #[serde(default)]
    pub contexts: Vec<ContextAssociation>,
    /// Most associated context
    pub primary_context: Option<String>,

    // === Sequence Information ===
    /// Typical position in sequence
    #[serde(default)]
    pub typical_position: u32,
    /// Frequently co-occurring phrases
    #[serde(default)]
    pub co_occurring_phrases: Vec<String>,

    // === Statistics ===
    /// Total occurrences in dataset
    #[serde(default)]
    pub occurrence_count: u32,
    /// Contribution to type entropy
    #[serde(default)]
    pub entropy_contribution: f64,

    // === Quality Metrics ===
    /// Recording quality
    #[serde(default)]
    pub signal_to_noise_ratio: f64,
    /// Confidence in extraction
    #[serde(default)]
    pub extraction_confidence: f64,

    // === Metadata ===
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Notes
    pub notes: Option<String>,
}

// Serialization helpers for species types
mod encoding_strategy_serde {
    use crate::species::EncodingStrategy;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(strategy: &EncodingStrategy, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match strategy {
            EncodingStrategy::Combinatorial => "combinatorial",
            EncodingStrategy::Quantitative => "quantitative",
            EncodingStrategy::CodaType => "coda_type",
            EncodingStrategy::FrequencyModulated => "frequency_modulated",
            EncodingStrategy::DurationMediated => "duration_mediated",
            EncodingStrategy::PhraseType => "phrase_type",
            EncodingStrategy::Minimal => "minimal",
        })
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<EncodingStrategy, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "combinatorial" => EncodingStrategy::Combinatorial,
            "quantitative" => EncodingStrategy::Quantitative,
            "coda_type" => EncodingStrategy::CodaType,
            "frequency_modulated" => EncodingStrategy::FrequencyModulated,
            "duration_mediated" => EncodingStrategy::DurationMediated,
            "phrase_type" => EncodingStrategy::PhraseType,
            _ => EncodingStrategy::Minimal,
        })
    }
}

mod encoding_modality_serde {
    use crate::species::AnalysisModality;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(modality: &AnalysisModality, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match modality {
            AnalysisModality::Temporal => "temporal",
            AnalysisModality::Spectral => "spectral",
            AnalysisModality::Hybrid => "hybrid",
        })
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<AnalysisModality, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "spectral" => AnalysisModality::Spectral,
            "hybrid" => AnalysisModality::Hybrid,
            _ => AnalysisModality::Temporal,
        })
    }
}

impl PhrasePrototype {
    /// Create new phrase prototype
    pub fn new(id: impl Into<String>, key: impl Into<String>, species: impl Into<String>) -> Self {
        Self {
            phrase_id: id.into(),
            phrase_key: key.into(),
            species: species.into(),
            source_file: None,
            source_dataset: None,
            encoding_strategy: crate::species::EncodingStrategy::PhraseType,
            encoding_modality: crate::species::AnalysisModality::Temporal,
            phrase_type: None,
            features_30d: AcousticFeatures30D::new(),
            contexts: Vec::new(),
            primary_context: None,
            typical_position: 0,
            co_occurring_phrases: Vec::new(),
            occurrence_count: 0,
            entropy_contribution: 0.0,
            signal_to_noise_ratio: 0.0,
            extraction_confidence: 0.0,
            created_at: Utc::now(),
            notes: None,
        }
    }

    /// Generate phrase key from features
    pub fn generate_key(&self, f0_bin_size: f64, dur_bin_size: f64) -> String {
        let f0_bin = (self.features_30d.mean_f0_hz / f0_bin_size).floor() * f0_bin_size;
        let dur_bin = (self.features_30d.duration_ms / dur_bin_size).floor() * dur_bin_size;
        format!("F0_{:.0}_DUR_{:.0}", f0_bin, dur_bin)
    }
}

// ============================================================================
// SPECIES PHRASE LIBRARY
// ============================================================================

/// Complete phrase library for a single species
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesPhraseLibrary {
    /// Species name
    pub species: String,
    /// Encoding strategy
    #[serde(with = "encoding_strategy_serde")]
    pub encoding_strategy: crate::species::EncodingStrategy,
    /// Encoding modality
    #[serde(with = "encoding_modality_serde")]
    pub encoding_modality: crate::species::AnalysisModality,

    // === Statistics ===
    /// Number of unique phrases
    pub total_phrases: usize,
    /// Total occurrences across all phrases
    pub total_occurrences: u64,
    /// Type entropy
    pub type_entropy: f64,
    /// Average phrases per file
    pub phrases_per_file_avg: f64,

    // === Phrase Collection ===
    /// All phrase prototypes
    pub phrases: Vec<PhrasePrototype>,

    // === Context Vocabulary ===
    /// All context labels
    #[serde(default)]
    pub context_labels: Vec<String>,

    // === Species Parameters ===
    /// Frequency range (min, max) in Hz
    pub frequency_range_hz: (f64, f64),
    /// Duration range (min, max) in ms
    pub typical_duration_ms: (f64, f64),

    // === Metadata ===
    /// Dataset information
    #[serde(default)]
    pub dataset_info: HashMap<String, String>,
    /// Extraction timestamp
    pub extraction_timestamp: DateTime<Utc>,
}

impl SpeciesPhraseLibrary {
    /// Create new empty library for a species
    pub fn new(species: impl Into<String>) -> Self {
        Self {
            species: species.into(),
            encoding_strategy: crate::species::EncodingStrategy::PhraseType,
            encoding_modality: crate::species::AnalysisModality::Temporal,
            total_phrases: 0,
            total_occurrences: 0,
            type_entropy: 0.0,
            phrases_per_file_avg: 0.0,
            phrases: Vec::new(),
            context_labels: Vec::new(),
            frequency_range_hz: (0.0, 0.0),
            typical_duration_ms: (0.0, 0.0),
            dataset_info: HashMap::new(),
            extraction_timestamp: Utc::now(),
        }
    }

    /// Add a phrase to the library
    pub fn add_phrase(&mut self, phrase: PhrasePrototype) {
        // Update context labels
        if let Some(ctx) = &phrase.primary_context {
            if !self.context_labels.contains(ctx) {
                self.context_labels.push(ctx.clone());
            }
        }

        // Update frequency range
        let f0 = phrase.features_30d.mean_f0_hz;
        if f0 > 0.0 {
            if self.frequency_range_hz.0 == 0.0 || f0 < self.frequency_range_hz.0 {
                self.frequency_range_hz.0 = f0;
            }
            if f0 > self.frequency_range_hz.1 {
                self.frequency_range_hz.1 = f0;
            }
        }

        // Update duration range
        let dur = phrase.features_30d.duration_ms;
        if dur > 0.0 {
            if self.typical_duration_ms.0 == 0.0 || dur < self.typical_duration_ms.0 {
                self.typical_duration_ms.0 = dur;
            }
            if dur > self.typical_duration_ms.1 {
                self.typical_duration_ms.1 = dur;
            }
        }

        self.phrases.push(phrase);
        self.total_phrases = self.phrases.len();
    }

    /// Recalculate statistics
    pub fn recalculate_statistics(&mut self) {
        self.total_occurrences = self.phrases.iter().map(|p| p.occurrence_count as u64).sum();

        // Calculate entropy
        if self.total_occurrences > 0 {
            let total = self.total_occurrences as f64;
            let entropy: f64 = self
                .phrases
                .iter()
                .map(|p| {
                    let p_prob = p.occurrence_count as f64 / total;
                    if p_prob > 0.0 {
                        -p_prob * p_prob.log2()
                    } else {
                        0.0
                    }
                })
                .sum();
            self.type_entropy = entropy;
        }
    }

    /// Find similar phrases
    pub fn find_similar(&self, query: &AcousticFeatures30D, threshold: f64) -> Vec<&PhrasePrototype> {
        self.phrases
            .iter()
            .filter(|p| p.features_30d.cosine_similarity(query) >= threshold)
            .collect()
    }

    /// Get phrases by context
    pub fn get_by_context(&self, context: &str) -> Vec<&PhrasePrototype> {
        self.phrases
            .iter()
            .filter(|p| p.primary_context.as_deref() == Some(context))
            .collect()
    }
}

// ============================================================================
// CROSS-SPECIES DATABASE
// ============================================================================

/// Unified phrase database across all species
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSpeciesPhraseDatabase {
    /// Species libraries
    #[serde(default)]
    pub species_libraries: HashMap<String, SpeciesPhraseLibrary>,

    // === Cross-species Analysis ===
    /// Species grouped by encoding strategy
    #[serde(default)]
    pub encoding_strategy_summary: HashMap<String, Vec<String>>,
    /// Species grouped by modality
    #[serde(default)]
    pub modality_summary: HashMap<String, Vec<String>>,

    // === Metadata ===
    /// Database version
    pub database_version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl CrossSpeciesPhraseDatabase {
    /// Create new empty database
    pub fn new() -> Self {
        Self {
            species_libraries: HashMap::new(),
            encoding_strategy_summary: HashMap::new(),
            modality_summary: HashMap::new(),
            database_version: "2.0".to_string(),
            created_at: Utc::now(),
        }
    }

    /// Add a species library
    pub fn add_library(&mut self, library: SpeciesPhraseLibrary) {
        let strategy = format!("{:?}", library.encoding_strategy).to_lowercase();
        let modality = format!("{:?}", library.encoding_modality).to_lowercase();
        let species = library.species.clone();

        // Update strategy summary
        self.encoding_strategy_summary
            .entry(strategy)
            .or_default()
            .push(species.clone());

        // Update modality summary
        self.modality_summary.entry(modality).or_default().push(species);

        // Add library
        self.species_libraries.insert(library.species.clone(), library);
    }

    /// Get total phrase count
    pub fn total_phrases(&self) -> usize {
        self.species_libraries.values().map(|l| l.total_phrases).sum()
    }

    /// Get total occurrence count
    pub fn total_occurrences(&self) -> u64 {
        self.species_libraries.values().map(|l| l.total_occurrences).sum()
    }

    /// Save to JSON file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> crate::ZooVoxResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: impl AsRef<std::path::Path>) -> crate::ZooVoxResult<Self> {
        let json = std::fs::read_to_string(path)?;
        let db: Self = serde_json::from_str(&json)?;
        Ok(db)
    }
}

impl Default for CrossSpeciesPhraseDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BEHAVIOR ANNOTATION
// ============================================================================

/// Behavioral annotation for audio segments
#[derive(Debug, Clone)]
pub struct BehaviorAnnotation {
    /// Start time in seconds
    pub start_seconds: f64,
    /// End time in seconds
    pub end_seconds: f64,
    /// Context label
    pub context_label: String,
    /// Context category
    pub context_category: String,
}

impl BehaviorAnnotation {
    /// Create new annotation
    pub fn new(start: f64, end: f64, label: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            start_seconds: start,
            end_seconds: end,
            context_label: label.into(),
            context_category: category.into(),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_30d_to_vector() {
        let features = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            f0_range_hz: 300.0,
            ..Default::default()
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), 30);
        assert!((vec[0] - 6800.0).abs() < 1e-10);
        assert!((vec[1] - 65.0).abs() < 1e-10);
    }

    #[test]
    fn test_features_30d_from_vector() {
        let mut vec = [0.0; 30];
        vec[0] = 7000.0;
        vec[1] = 50.0;
        vec[13] = -500.0; // MFCC 1

        let features = AcousticFeatures30D::from_vector(vec);
        assert!((features.mean_f0_hz - 7000.0).abs() < 1e-10);
        assert!((features.duration_ms - 50.0).abs() < 1e-10);
        assert!((features.mfcc_1 - (-500.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_30d() {
        let f1 = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let f2 = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        // Identical features should have similarity 1.0
        let sim = f1.cosine_similarity(&f2);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    // ========================================================================
    // 45D FEATURE TESTS (TDD)
    // ========================================================================

    #[test]
    fn test_features_45d_to_vector() {
        let features = AcousticFeatures45D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            f0_range_hz: 300.0,
            // Resonance (NEW)
            formant_1_hz: 500.0,
            formant_2_hz: 1500.0,
            formant_3_hz: 2500.0,
            formant_1_bandwidth: 80.0,
            formant_2_bandwidth: 100.0,
            formant_dispersion: 1000.0,
            // Spectral Shape (NEW)
            spectral_centroid: 2000.0,
            spectral_spread: 1500.0,
            spectral_skewness: 0.5,
            spectral_kurtosis: 3.0,
            // Modulation (NEW)
            spectral_tilt: -6.0,
            fm_slope_hz_per_sec: 5000.0,
            am_depth: 0.3,
            // Non-Linear (NEW)
            subharmonic_ratio: 0.1,
            spectral_entropy: 0.5,
            ..Default::default()
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), 45, "45D vector should have 45 elements");
        assert!((vec[0] - 6800.0).abs() < 1e-10, "Fundamental F0");
        assert!((vec[1] - 65.0).abs() < 1e-10, "Duration");

        // Check Resonance factors (indices 30-35)
        assert!((vec[30] - 500.0).abs() < 1e-10, "Formant 1 Hz");
        assert!((vec[31] - 1500.0).abs() < 1e-10, "Formant 2 Hz");
        assert!((vec[32] - 2500.0).abs() < 1e-10, "Formant 3 Hz");
        assert!((vec[33] - 80.0).abs() < 1e-10, "Formant 1 Bandwidth");
        assert!((vec[34] - 100.0).abs() < 1e-10, "Formant 2 Bandwidth");
        assert!((vec[35] - 1000.0).abs() < 1e-10, "Formant Dispersion");

        // Check Spectral Shape factors (indices 36-39)
        assert!((vec[36] - 2000.0).abs() < 1e-10, "Spectral Centroid");
        assert!((vec[37] - 1500.0).abs() < 1e-10, "Spectral Spread");
        assert!((vec[38] - 0.5).abs() < 1e-10, "Spectral Skewness");
        assert!((vec[39] - 3.0).abs() < 1e-10, "Spectral Kurtosis");

        // Check Modulation factors (indices 40-42)
        assert!((vec[40] - (-6.0)).abs() < 1e-10, "Spectral Tilt");
        assert!((vec[41] - 5000.0).abs() < 1e-10, "FM Slope");
        assert!((vec[42] - 0.3).abs() < 1e-10, "AM Depth");

        // Check Non-Linear factors (indices 43-44)
        assert!((vec[43] - 0.1).abs() < 1e-10, "Subharmonic Ratio");
        assert!((vec[44] - 0.5).abs() < 1e-10, "Spectral Entropy");
    }

    #[test]
    fn test_features_45d_from_vector() {
        let mut vec = [0.0; 45];
        vec[0] = 7000.0; // F0
        vec[1] = 50.0; // Duration
        vec[30] = 600.0; // Formant 1
        vec[36] = 3000.0; // Spectral Centroid
        vec[40] = -12.0; // Spectral Tilt
        vec[43] = 0.2; // Subharmonic Ratio

        let features = AcousticFeatures45D::from_vector(vec);
        assert!((features.mean_f0_hz - 7000.0).abs() < 1e-10);
        assert!((features.duration_ms - 50.0).abs() < 1e-10);
        assert!((features.formant_1_hz - 600.0).abs() < 1e-10);
        assert!((features.spectral_centroid - 3000.0).abs() < 1e-10);
        assert!((features.spectral_tilt - (-12.0)).abs() < 1e-10);
        assert!((features.subharmonic_ratio - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_features_45d_cosine_similarity() {
        let f1 = AcousticFeatures45D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            formant_1_hz: 500.0,
            spectral_centroid: 2000.0,
            spectral_tilt: -6.0,
            ..Default::default()
        };

        let f2 = AcousticFeatures45D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            formant_1_hz: 500.0,
            spectral_centroid: 2000.0,
            spectral_tilt: -6.0,
            ..Default::default()
        };

        // Identical features should have similarity 1.0
        let sim = f1.cosine_similarity(&f2);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_features_45d_distance() {
        let f1 = AcousticFeatures45D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let f2 = AcousticFeatures45D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        // Identical features should have distance 0.0
        let dist = f1.distance(&f2);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_features_45d_from_30d() {
        let features_30d = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            f0_range_hz: 300.0,
            harmonic_to_noise_ratio: 15.0,
            spectral_flatness: 0.3,
            ..Default::default()
        };

        let features_45d = AcousticFeatures45D::from_30d(&features_30d);

        // Check 30D values are preserved
        assert!((features_45d.mean_f0_hz - 6800.0).abs() < 1e-10);
        assert!((features_45d.duration_ms - 65.0).abs() < 1e-10);
        assert!((features_45d.f0_range_hz - 300.0).abs() < 1e-10);
        assert!((features_45d.harmonic_to_noise_ratio - 15.0).abs() < 1e-10);
        assert!((features_45d.spectral_flatness - 0.3).abs() < 1e-10);

        // Check new 15D fields are zero
        assert!((features_45d.formant_1_hz).abs() < 1e-10);
        assert!((features_45d.spectral_centroid).abs() < 1e-10);
        assert!((features_45d.spectral_tilt).abs() < 1e-10);
        assert!((features_45d.subharmonic_ratio).abs() < 1e-10);
    }

    #[test]
    fn test_features_45d_serialization() {
        let features = AcousticFeatures45D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            formant_1_hz: 500.0,
            spectral_centroid: 2000.0,
            spectral_tilt: -6.0,
            subharmonic_ratio: 0.1,
            ..Default::default()
        };

        let json = serde_json::to_string(&features).unwrap();
        let decoded: AcousticFeatures45D = serde_json::from_str(&json).unwrap();

        assert!((decoded.mean_f0_hz - 6800.0).abs() < 1e-10);
        assert!((decoded.formant_1_hz - 500.0).abs() < 1e-10);
        assert!((decoded.spectral_centroid - 2000.0).abs() < 1e-10);
        assert!((decoded.spectral_tilt - (-6.0)).abs() < 1e-10);
        assert!((decoded.subharmonic_ratio - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_phrase_prototype() {
        let phrase = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        assert_eq!(phrase.phrase_id, "marmoset_001");
        assert_eq!(phrase.species, "marmoset");
    }

    #[test]
    fn test_species_library() {
        let mut library = SpeciesPhraseLibrary::new("marmoset");

        let phrase = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        library.add_phrase(phrase);

        assert_eq!(library.total_phrases, 1);
    }

    #[test]
    fn test_serialization_30d() {
        let features = AcousticFeatures30D {
            mean_f0_hz: 6800.0,
            duration_ms: 65.0,
            ..Default::default()
        };

        let json = serde_json::to_string(&features).unwrap();
        let decoded: AcousticFeatures30D = serde_json::from_str(&json).unwrap();

        assert!((decoded.mean_f0_hz - 6800.0).abs() < 1e-10);
    }
}
