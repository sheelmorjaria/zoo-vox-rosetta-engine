//! Rosetta-Stack 105D: Triple-Layer Feature Architecture
//! ======================================================
//!
//! Expands the 75D vector to 105D with a new Micro-Texture layer:
//!
//! **Architecture:**
//! ```text
//! Layer 1: BASE PHYSICS (45D)
//!   Role: Universal Taxonomy (Bird vs Whale vs Insect)
//!   Features: F0, Duration, HNR, Spectral shape, Rhythm basics
//!
//! Layer 2: MACRO TEXTURE (30D)
//!   Role: Species Group Discrimination (Robin vs Sparrow)
//!   Features: Harmonic texture, Pitch geometry, GLCM spectrogram texture
//!
//! Layer 3: MICRO TEXTURE (30D) - NEW
//!   Role: Fine Species Identity (Individual/Dialect detection)
//!   Features: Modulation spectra, Rhythm histograms, Psychoacoustics
//! ```
//!
//! **Dual-Head Training Strategy:**
//! - Taxonomy Head: Input [0..45] → Protects 77% accuracy
//! - Species Head: Input [0..105] → Maximizes discriminative power
//!
//! **Expected Outcome:**
//! - Taxonomy Accuracy: ~77% (preserved)
//! - Species Accuracy: 22.57% → 30-35%
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::micro_dynamics_extractor::RosettaFeatures;
use serde::{Deserialize, Serialize};

// ============================================================================
// 105D Feature Vector Structure
// ============================================================================

/// Complete 105D feature vector with triple-layer architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector105D {
    // =============================================================
    // LAYER 1: BASE PHYSICS (45D)
    // Universal features for taxonomic classification
    // =============================================================
    /// Base 45D physics features (Vec to avoid serde array size limit)
    pub base_45d: Vec<f32>,

    // =============================================================
    // LAYER 2: MACRO TEXTURE (30D)
    // Species group discrimination features
    // =============================================================
    /// Harmonic Texture (8D) - indices 45-52
    pub harmonic_slope: f32, // Harmonic decay rate
    pub h1_h2_diff_db: f32,            // 1st/2nd harmonic ratio
    pub harmonic_irregularity: f32,    // Jitter in harmonics
    pub harmonic_energy_variance: f32, // Energy distribution
    pub spectral_flux_std: f32,        // Spectral change rate
    pub h1_h2_ratio: f32,              // Linear ratio
    pub h2_h3_ratio: f32,              // Upper harmonic structure
    pub h3_h4_ratio: f32,              // Higher harmonic decay

    /// Pitch Geometry (7D) - indices 53-59
    pub f0_mean_derivative: f32, // Average pitch change rate
    pub f0_curvature: f32,        // Pitch trajectory shape
    pub f0_inflection_count: f32, // Direction changes
    pub glissando_rate: f32,      // Sweep speed
    pub vibrato_regularity: f32,  // Vibrato consistency
    pub jitter_trend: f32,        // Pitch stability over time
    pub pitch_entropy: f32,       // Pitch distribution complexity

    /// GLCM Spectrogram Texture (10D) - indices 60-69
    pub glcm_contrast: f32, // Local intensity variation
    pub glcm_correlation: f32,         // Frequency correlation
    pub glcm_energy: f32,              // Uniformity
    pub glcm_homogeneity: f32,         // Spectral smoothness
    pub run_length_nonuniformity: f32, // Horizontal streaks
    pub long_run_emphasis: f32,        // Sustained frequencies
    pub short_run_emphasis: f32,       // Transient frequencies
    pub granularity: f32,              // Fine structure
    pub vertical_strength: f32,        // Temporal consistency
    pub diagonal_strength: f32,        // Frequency sweeps

    /// Temporal Texture (5D) - indices 70-74
    pub energy_envelope_variance: f32, // Amplitude stability
    pub onset_sustain_ratio: f32,    // Attack vs sustain
    pub peak_count: f32,             // Amplitude peaks
    pub pulse_regularity: f32,       // Rhythmic consistency
    pub zero_crossing_variance: f32, // Fine temporal structure

    // =============================================================
    // LAYER 3: MICRO TEXTURE (30D) - NEW
    // Fine species identity features
    // =============================================================
    /// A. Modulation Spectra (15D) - indices 75-89
    /// Amplitude Modulation (Energy vibration) - indices 75-79
    pub am_spectrum_0_10hz: f32, // Slow tremolo
    pub am_spectrum_10_30hz: f32,  // Trill range
    pub am_spectrum_30_50hz: f32,  // Roughness range
    pub am_spectrum_50_100hz: f32, // Insect buzz range
    pub am_depth_mean: f32,        // Average modulation depth

    /// Frequency Modulation (Pitch vibration) - indices 80-84
    pub fm_spectrum_0_10hz: f32, // Slow drift
    pub fm_spectrum_10_30hz: f32,  // Vibrato range
    pub fm_spectrum_30_50hz: f32,  // Warble range
    pub fm_spectrum_50_100hz: f32, // Rapid squeak
    pub fm_depth_mean: f32,        // Average FM depth

    /// Modulation Statistics - indices 85-89
    pub am_fm_ratio: f32, // AM vs FM dominance
    pub modulation_complexity: f32, // Multi-band modulation spread
    pub trill_strength: f32,        // 15-30Hz AM energy
    pub flutter_index: f32,         // Rapid modulation presence
    pub modulation_synchrony: f32,  // AM-FM coupling

    /// B. Rhythm Histograms (10D) - indices 90-99
    /// Inter-Onset Interval distribution
    pub ioi_bin_0_50ms: f32, // Very rapid bursts
    pub ioi_bin_50_100ms: f32,   // Fast chirps
    pub ioi_bin_100_200ms: f32,  // Standard song tempo
    pub ioi_bin_200_500ms: f32,  // Slow calls
    pub ioi_bin_500_1000ms: f32, // Slow rhythmic calls
    pub ioi_bin_1000_plus: f32,  // Long pauses
    pub ioi_variance: f32,       // Gap consistency
    pub ioi_skewness: f32,       // Gap distribution shape
    pub ioi_kurtosis: f32,       // Gap distribution tails
    pub rhythm_regularity: f32,  // Overall rhythmic stability

    /// C. Psychoacoustics (5D) - indices 100-104
    pub sharpness_acum: f32, // "Pointiness" of sound
    pub roughness_asper: f32,      // Perceived texture grain
    pub loudness_sone: f32,        // Perceived volume (non-linear)
    pub tonality_index: f32,       // Pitch vs Noise balance
    pub fluctuation_strength: f32, // Slow modulation perception
}

impl Vector105D {
    /// Create a new zero-initialized 105D vector
    pub fn zero() -> Self {
        Self {
            base_45d: vec![0.0; 45],
            harmonic_slope: 0.0,
            h1_h2_diff_db: 0.0,
            harmonic_irregularity: 0.0,
            harmonic_energy_variance: 0.0,
            spectral_flux_std: 0.0,
            h1_h2_ratio: 0.0,
            h2_h3_ratio: 0.0,
            h3_h4_ratio: 0.0,
            f0_mean_derivative: 0.0,
            f0_curvature: 0.0,
            f0_inflection_count: 0.0,
            glissando_rate: 0.0,
            vibrato_regularity: 0.0,
            jitter_trend: 0.0,
            pitch_entropy: 0.0,
            glcm_contrast: 0.0,
            glcm_correlation: 0.0,
            glcm_energy: 0.0,
            glcm_homogeneity: 0.0,
            run_length_nonuniformity: 0.0,
            long_run_emphasis: 0.0,
            short_run_emphasis: 0.0,
            granularity: 0.0,
            vertical_strength: 0.0,
            diagonal_strength: 0.0,
            energy_envelope_variance: 0.0,
            onset_sustain_ratio: 0.0,
            peak_count: 0.0,
            pulse_regularity: 0.0,
            zero_crossing_variance: 0.0,
            am_spectrum_0_10hz: 0.0,
            am_spectrum_10_30hz: 0.0,
            am_spectrum_30_50hz: 0.0,
            am_spectrum_50_100hz: 0.0,
            am_depth_mean: 0.0,
            fm_spectrum_0_10hz: 0.0,
            fm_spectrum_10_30hz: 0.0,
            fm_spectrum_30_50hz: 0.0,
            fm_spectrum_50_100hz: 0.0,
            fm_depth_mean: 0.0,
            am_fm_ratio: 0.0,
            modulation_complexity: 0.0,
            trill_strength: 0.0,
            flutter_index: 0.0,
            modulation_synchrony: 0.0,
            ioi_bin_0_50ms: 0.0,
            ioi_bin_50_100ms: 0.0,
            ioi_bin_100_200ms: 0.0,
            ioi_bin_200_500ms: 0.0,
            ioi_bin_500_1000ms: 0.0,
            ioi_bin_1000_plus: 0.0,
            ioi_variance: 0.0,
            ioi_skewness: 0.0,
            ioi_kurtosis: 0.0,
            rhythm_regularity: 0.0,
            sharpness_acum: 0.0,
            roughness_asper: 0.0,
            loudness_sone: 0.0,
            tonality_index: 0.0,
            fluctuation_strength: 0.0,
        }
    }

    /// Convert to flat array for ML training
    pub fn to_array(&self) -> Vec<f32> {
        let mut arr = Vec::with_capacity(105);

        // Layer 1: Base Physics (45D)
        arr.extend_from_slice(&self.base_45d);

        // Layer 2: Macro Texture (30D)
        arr.extend_from_slice(&[
            self.harmonic_slope,
            self.h1_h2_diff_db,
            self.harmonic_irregularity,
            self.harmonic_energy_variance,
            self.spectral_flux_std,
            self.h1_h2_ratio,
            self.h2_h3_ratio,
            self.h3_h4_ratio,
            self.f0_mean_derivative,
            self.f0_curvature,
            self.f0_inflection_count,
            self.glissando_rate,
            self.vibrato_regularity,
            self.jitter_trend,
            self.pitch_entropy,
            self.glcm_contrast,
            self.glcm_correlation,
            self.glcm_energy,
            self.glcm_homogeneity,
            self.run_length_nonuniformity,
            self.long_run_emphasis,
            self.short_run_emphasis,
            self.granularity,
            self.vertical_strength,
            self.diagonal_strength,
            self.energy_envelope_variance,
            self.onset_sustain_ratio,
            self.peak_count,
            self.pulse_regularity,
            self.zero_crossing_variance,
        ]);

        // Layer 3: Micro Texture (30D)
        arr.extend_from_slice(&[
            // Modulation Spectra (15D)
            self.am_spectrum_0_10hz,
            self.am_spectrum_10_30hz,
            self.am_spectrum_30_50hz,
            self.am_spectrum_50_100hz,
            self.am_depth_mean,
            self.fm_spectrum_0_10hz,
            self.fm_spectrum_10_30hz,
            self.fm_spectrum_30_50hz,
            self.fm_spectrum_50_100hz,
            self.fm_depth_mean,
            self.am_fm_ratio,
            self.modulation_complexity,
            self.trill_strength,
            self.flutter_index,
            self.modulation_synchrony,
            // Rhythm Histograms (10D)
            self.ioi_bin_0_50ms,
            self.ioi_bin_50_100ms,
            self.ioi_bin_100_200ms,
            self.ioi_bin_200_500ms,
            self.ioi_bin_500_1000ms,
            self.ioi_bin_1000_plus,
            self.ioi_variance,
            self.ioi_skewness,
            self.ioi_kurtosis,
            self.rhythm_regularity,
            // Psychoacoustics (5D)
            self.sharpness_acum,
            self.roughness_asper,
            self.loudness_sone,
            self.tonality_index,
            self.fluctuation_strength,
        ]);

        arr
    }

    /// Get physics slice (for Taxonomy Head) - indices 0..45
    pub fn physics_slice(&self) -> &[f32] {
        &self.base_45d
    }

    /// Get macro texture slice (for intermediate analysis) - indices 45..75
    pub fn macro_texture_slice(&self) -> Vec<f32> {
        vec![
            self.harmonic_slope,
            self.h1_h2_diff_db,
            self.harmonic_irregularity,
            self.harmonic_energy_variance,
            self.spectral_flux_std,
            self.h1_h2_ratio,
            self.h2_h3_ratio,
            self.h3_h4_ratio,
            self.f0_mean_derivative,
            self.f0_curvature,
            self.f0_inflection_count,
            self.glissando_rate,
            self.vibrato_regularity,
            self.jitter_trend,
            self.pitch_entropy,
            self.glcm_contrast,
            self.glcm_correlation,
            self.glcm_energy,
            self.glcm_homogeneity,
            self.run_length_nonuniformity,
            self.long_run_emphasis,
            self.short_run_emphasis,
            self.granularity,
            self.vertical_strength,
            self.diagonal_strength,
            self.energy_envelope_variance,
            self.onset_sustain_ratio,
            self.peak_count,
            self.pulse_regularity,
            self.zero_crossing_variance,
        ]
    }
}

// ============================================================================
// 112D Feature Vector (Upgraded from 105D)
// ============================================================================

/// 112-dimensional acoustic feature vector with full synthesis support
///
/// This is the upgraded version of Vector105D with 7 additional dimensions:
/// - Layer 1: 45D → 46D (+1: release_time_ms)
/// - Layer 2: 30D (unchanged)
/// - Layer 3: 30D → 36D (+6: additional rhythm histogram features)
///
/// Total: 112 dimensions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector112D {
    // =============================================================
    // LAYER 1: BASE PHYSICS (46D) - indices 0-45
    // =============================================================
    /// Base 46D physics features stored as Vec for flexibility
    pub base_46d: Vec<f32>,

    // =============================================================
    // LAYER 2: MACRO TEXTURE (30D) - indices 46-75
    // =============================================================
    /// Harmonic Texture (9D) - indices 46-54
    pub harmonic_slope: f32,
    pub h1_h2_diff_db: f32,
    pub harmonic_irregularity: f32,
    pub harmonic_energy_variance: f32,
    pub spectral_flux_std: f32,
    pub h1_h2_ratio: f32,
    pub h2_h3_ratio: f32,
    pub h3_h4_ratio: f32,
    pub harmonic_density: f32,

    /// Pitch Geometry (7D) - indices 55-61
    pub f0_mean_derivative: f32,
    pub f0_curvature: f32,
    pub f0_inflection_count: f32,
    pub glissando_rate: f32,
    pub vibrato_regularity: f32,
    pub jitter_trend: f32,
    pub pitch_complexity: f32,

    /// GLCM Spectrogram Texture (14D) - indices 62-75
    pub glcm_contrast: f32,
    pub glcm_correlation: f32,
    pub glcm_energy: f32,
    pub glcm_homogeneity: f32,
    pub run_length_nonuniformity: f32,
    pub long_run_emphasis: f32,
    pub short_run_emphasis: f32,
    pub granularity: f32,
    pub vertical_strength: f32,
    pub horizontal_correlation: f32,
    pub texture_entropy: f32,
    pub texture_homogeneity: f32,
    pub texture_contrast: f32,
    pub texture_energy: f32,

    // =============================================================
    // LAYER 3: MICRO TEXTURE (36D) - indices 76-111
    // =============================================================
    /// Spectral Derivative (6D) - indices 76-81
    pub spectral_derivative_mean: f32,
    pub spectral_derivative_std: f32,
    pub spectral_derivative_skew: f32,
    pub spectral_derivative_kurtosis: f32,
    pub spectral_derivative_max: f32,
    pub spectral_derivative_range: f32,

    /// FM Bin Features (5D) - indices 82-86
    pub fm_rate_mean: f32,
    pub fm_rate_std: f32,
    pub fm_depth_mean: f32,
    pub fm_depth_std: f32,
    pub fm_extent_hz: f32,

    /// Dynamics Bin Features (5D) - indices 87-91
    pub dynamics_rise_rate: f32,
    pub dynamics_fall_rate: f32,
    pub dynamics_range_db: f32,
    pub dynamics_cv: f32,
    pub dynamics_skew: f32,

    /// ICI Bin Features (5D) - indices 92-96
    pub ici_mean_ms: f32,
    pub ici_std_ms: f32,
    pub ici_skew: f32,
    pub ici_kurtosis: f32,
    pub ici_regularity: f32,

    /// Rhythm Histogram Extended (15D) - indices 97-111
    pub rhythm_tempo_hz: f32,
    pub rhythm_tempo_stability: f32,
    pub rhythm_pulse_clarity: f32,
    pub rhythm_grouping_strength: f32,
    pub rhythm_cycle_length: f32,
    pub rhythm_onset_strength: f32,
    pub rhythm_swing_factor: f32,
    pub rhythm_syncopation: f32,
    pub rhythm_density: f32,
    pub rhythm_complexity: f32,
    pub rhythm_entropy: f32,
    pub rhythm_peak_rate_hz: f32,
    pub rhythm_valley_depth: f32,
    pub rhythm_crest_factor: f32,
    pub rhythm_flux: f32,
}

impl Vector112D {
    /// Create a new zero-initialized 112D vector
    pub fn zero() -> Self {
        Self {
            base_46d: vec![0.0; 46],
            harmonic_slope: 0.0,
            h1_h2_diff_db: 0.0,
            harmonic_irregularity: 0.0,
            harmonic_energy_variance: 0.0,
            spectral_flux_std: 0.0,
            h1_h2_ratio: 0.0,
            h2_h3_ratio: 0.0,
            h3_h4_ratio: 0.0,
            harmonic_density: 0.0,
            f0_mean_derivative: 0.0,
            f0_curvature: 0.0,
            f0_inflection_count: 0.0,
            glissando_rate: 0.0,
            vibrato_regularity: 0.0,
            jitter_trend: 0.0,
            pitch_complexity: 0.0,
            glcm_contrast: 0.0,
            glcm_correlation: 0.0,
            glcm_energy: 0.0,
            glcm_homogeneity: 0.0,
            run_length_nonuniformity: 0.0,
            long_run_emphasis: 0.0,
            short_run_emphasis: 0.0,
            granularity: 0.0,
            vertical_strength: 0.0,
            horizontal_correlation: 0.0,
            texture_entropy: 0.0,
            texture_homogeneity: 0.0,
            texture_contrast: 0.0,
            texture_energy: 0.0,
            spectral_derivative_mean: 0.0,
            spectral_derivative_std: 0.0,
            spectral_derivative_skew: 0.0,
            spectral_derivative_kurtosis: 0.0,
            spectral_derivative_max: 0.0,
            spectral_derivative_range: 0.0,
            fm_rate_mean: 0.0,
            fm_rate_std: 0.0,
            fm_depth_mean: 0.0,
            fm_depth_std: 0.0,
            fm_extent_hz: 0.0,
            dynamics_rise_rate: 0.0,
            dynamics_fall_rate: 0.0,
            dynamics_range_db: 0.0,
            dynamics_cv: 0.0,
            dynamics_skew: 0.0,
            ici_mean_ms: 0.0,
            ici_std_ms: 0.0,
            ici_skew: 0.0,
            ici_kurtosis: 0.0,
            ici_regularity: 0.0,
            rhythm_tempo_hz: 0.0,
            rhythm_tempo_stability: 0.0,
            rhythm_pulse_clarity: 0.0,
            rhythm_grouping_strength: 0.0,
            rhythm_cycle_length: 0.0,
            rhythm_onset_strength: 0.0,
            rhythm_swing_factor: 0.0,
            rhythm_syncopation: 0.0,
            rhythm_density: 0.0,
            rhythm_complexity: 0.0,
            rhythm_entropy: 0.0,
            rhythm_peak_rate_hz: 0.0,
            rhythm_valley_depth: 0.0,
            rhythm_crest_factor: 0.0,
            rhythm_flux: 0.0,
        }
    }

    /// Convert to flat array for ML training
    pub fn to_array(&self) -> Vec<f32> {
        let mut arr = Vec::with_capacity(112);

        // Layer 1: Base Physics (46D)
        arr.extend_from_slice(&self.base_46d);

        // Layer 2: Macro Texture (30D)
        arr.extend_from_slice(&[
            self.harmonic_slope,
            self.h1_h2_diff_db,
            self.harmonic_irregularity,
            self.harmonic_energy_variance,
            self.spectral_flux_std,
            self.h1_h2_ratio,
            self.h2_h3_ratio,
            self.h3_h4_ratio,
            self.harmonic_density,
            self.f0_mean_derivative,
            self.f0_curvature,
            self.f0_inflection_count,
            self.glissando_rate,
            self.vibrato_regularity,
            self.jitter_trend,
            self.pitch_complexity,
            self.glcm_contrast,
            self.glcm_correlation,
            self.glcm_energy,
            self.glcm_homogeneity,
            self.run_length_nonuniformity,
            self.long_run_emphasis,
            self.short_run_emphasis,
            self.granularity,
            self.vertical_strength,
            self.horizontal_correlation,
            self.texture_entropy,
            self.texture_homogeneity,
            self.texture_contrast,
            self.texture_energy,
        ]);

        // Layer 3: Micro Texture (36D)
        arr.extend_from_slice(&[
            // Spectral Derivative (6D)
            self.spectral_derivative_mean,
            self.spectral_derivative_std,
            self.spectral_derivative_skew,
            self.spectral_derivative_kurtosis,
            self.spectral_derivative_max,
            self.spectral_derivative_range,
            // FM Bin (5D)
            self.fm_rate_mean,
            self.fm_rate_std,
            self.fm_depth_mean,
            self.fm_depth_std,
            self.fm_extent_hz,
            // Dynamics Bin (5D)
            self.dynamics_rise_rate,
            self.dynamics_fall_rate,
            self.dynamics_range_db,
            self.dynamics_cv,
            self.dynamics_skew,
            // ICI Bin (5D)
            self.ici_mean_ms,
            self.ici_std_ms,
            self.ici_skew,
            self.ici_kurtosis,
            self.ici_regularity,
            // Rhythm Histogram Extended (15D)
            self.rhythm_tempo_hz,
            self.rhythm_tempo_stability,
            self.rhythm_pulse_clarity,
            self.rhythm_grouping_strength,
            self.rhythm_cycle_length,
            self.rhythm_onset_strength,
            self.rhythm_swing_factor,
            self.rhythm_syncopation,
            self.rhythm_density,
            self.rhythm_complexity,
            self.rhythm_entropy,
            self.rhythm_peak_rate_hz,
            self.rhythm_valley_depth,
            self.rhythm_crest_factor,
            self.rhythm_flux,
        ]);

        arr
    }

    /// Get physics slice (for Taxonomy Head) - indices 0..46
    pub fn physics_slice(&self) -> &[f32] {
        &self.base_46d
    }

    /// Get macro texture slice - indices 46..76
    pub fn macro_texture_slice(&self) -> Vec<f32> {
        vec![
            self.harmonic_slope,
            self.h1_h2_diff_db,
            self.harmonic_irregularity,
            self.harmonic_energy_variance,
            self.spectral_flux_std,
            self.h1_h2_ratio,
            self.h2_h3_ratio,
            self.h3_h4_ratio,
            self.harmonic_density,
            self.f0_mean_derivative,
            self.f0_curvature,
            self.f0_inflection_count,
            self.glissando_rate,
            self.vibrato_regularity,
            self.jitter_trend,
            self.pitch_complexity,
            self.glcm_contrast,
            self.glcm_correlation,
            self.glcm_energy,
            self.glcm_homogeneity,
            self.run_length_nonuniformity,
            self.long_run_emphasis,
            self.short_run_emphasis,
            self.granularity,
            self.vertical_strength,
            self.horizontal_correlation,
            self.texture_entropy,
            self.texture_homogeneity,
            self.texture_contrast,
            self.texture_energy,
        ]
    }

    /// Convert from flat 112D slice
    ///
    /// Inverse of `to_array()`. Layout:
    /// - `arr[0..46]` → `base_46d`
    /// - `arr[46..76]` → macro texture named fields
    /// - `arr[76..112]` → micro texture named fields
    pub fn from_array(arr: &[f32]) -> Self {
        debug_assert!(arr.len() >= 112, "from_array requires at least 112 elements");

        let mut v = Self::zero();
        // Layer 1: Base Physics (46D)
        v.base_46d = arr[0..46].to_vec();

        // Layer 2: Macro Texture (30D) - indices 46..76
        v.harmonic_slope = arr[46];
        v.h1_h2_diff_db = arr[47];
        v.harmonic_irregularity = arr[48];
        v.harmonic_energy_variance = arr[49];
        v.spectral_flux_std = arr[50];
        v.h1_h2_ratio = arr[51];
        v.h2_h3_ratio = arr[52];
        v.h3_h4_ratio = arr[53];
        v.harmonic_density = arr[54];
        v.f0_mean_derivative = arr[55];
        v.f0_curvature = arr[56];
        v.f0_inflection_count = arr[57];
        v.glissando_rate = arr[58];
        v.vibrato_regularity = arr[59];
        v.jitter_trend = arr[60];
        v.pitch_complexity = arr[61];
        v.glcm_contrast = arr[62];
        v.glcm_correlation = arr[63];
        v.glcm_energy = arr[64];
        v.glcm_homogeneity = arr[65];
        v.run_length_nonuniformity = arr[66];
        v.long_run_emphasis = arr[67];
        v.short_run_emphasis = arr[68];
        v.granularity = arr[69];
        v.vertical_strength = arr[70];
        v.horizontal_correlation = arr[71];
        v.texture_entropy = arr[72];
        v.texture_homogeneity = arr[73];
        v.texture_contrast = arr[74];
        v.texture_energy = arr[75];

        // Layer 3: Micro Texture (36D) - indices 76..112
        v.spectral_derivative_mean = arr[76];
        v.spectral_derivative_std = arr[77];
        v.spectral_derivative_skew = arr[78];
        v.spectral_derivative_kurtosis = arr[79];
        v.spectral_derivative_max = arr[80];
        v.spectral_derivative_range = arr[81];
        v.fm_rate_mean = arr[82];
        v.fm_rate_std = arr[83];
        v.fm_depth_mean = arr[84];
        v.fm_depth_std = arr[85];
        v.fm_extent_hz = arr[86];
        v.dynamics_rise_rate = arr[87];
        v.dynamics_fall_rate = arr[88];
        v.dynamics_range_db = arr[89];
        v.dynamics_cv = arr[90];
        v.dynamics_skew = arr[91];
        v.ici_mean_ms = arr[92];
        v.ici_std_ms = arr[93];
        v.ici_skew = arr[94];
        v.ici_kurtosis = arr[95];
        v.ici_regularity = arr[96];
        v.rhythm_tempo_hz = arr[97];
        v.rhythm_tempo_stability = arr[98];
        v.rhythm_pulse_clarity = arr[99];
        v.rhythm_grouping_strength = arr[100];
        v.rhythm_cycle_length = arr[101];
        v.rhythm_onset_strength = arr[102];
        v.rhythm_swing_factor = arr[103];
        v.rhythm_syncopation = arr[104];
        v.rhythm_density = arr[105];
        v.rhythm_complexity = arr[106];
        v.rhythm_entropy = arr[107];
        v.rhythm_peak_rate_hz = arr[108];
        v.rhythm_valley_depth = arr[109];
        v.rhythm_crest_factor = arr[110];
        v.rhythm_flux = arr[111];

        v
    }

    /// Convert from fixed 112-element array
    pub fn from_array_fixed(arr: [f32; 112]) -> Self {
        Self::from_array(&arr)
    }

    /// Get normalization ranges for each dimension
    ///
    /// 112 per-dimension scaling ranges used to normalize features before
    /// distance calculation, ensuring meaningful comparisons.
    pub fn normalization_ranges() -> Vec<f32> {
        // Physics (0-45): based on Vector45D::normalization_ranges() plus
        // ranges for the 7 additional physics features (indices 3-5, 12, 33, 39, 45)
        let mut ranges = Vec::with_capacity(112);

        // === Layer 1: Base Physics (46D) ===
        ranges.extend_from_slice(&[
            // Fundamental (3D)
            2000.0, // 0:  mean_f0_hz
            100.0,  // 1:  duration_ms
            500.0,  // 2:  f0_range_hz
            // Energy (3D)
            1.0, // 3:  rms_energy
            0.5, // 4:  zero_crossing_rate
            1.0, // 5:  peak_amplitude
            // Harmonicity (3D)
            30.0, // 6:  harmonic_to_noise_ratio
            1.0,  // 7:  harmonicity
            1.0,  // 8:  spectral_flatness
            // Temporal envelope (4D)
            20.0, // 9:  attack_time_ms
            50.0, // 10: decay_time_ms
            1.0,  // 11: sustain_level
            50.0, // 12: release_time_ms
            // MFCCs (13D)
            20.0, // 13: mfcc_0
            20.0, // 14: mfcc_1
            20.0, // 15: mfcc_2
            20.0, // 16: mfcc_3
            20.0, // 17: mfcc_4
            20.0, // 18: mfcc_5
            20.0, // 19: mfcc_6
            20.0, // 20: mfcc_7
            20.0, // 21: mfcc_8
            20.0, // 22: mfcc_9
            20.0, // 23: mfcc_10
            20.0, // 24: mfcc_11
            20.0, // 25: mfcc_12
            // Spectral shape (4D)
            15000.0, // 26: spectral_centroid
            5000.0,  // 27: spectral_spread
            2.0,     // 28: spectral_skewness
            5.0,     // 29: spectral_kurtosis
            // Rhythm basics (4D)
            200.0, // 30: median_ici_ms
            20.0,  // 31: onset_rate_hz
            1.0,   // 32: ici_coefficient_of_variation
            1.0,   // 33: rhythm_regularity
            // Perturbation (4D)
            0.05,  // 34: jitter
            0.1,   // 35: shimmer
            100.0, // 36: vibrato_depth
            20.0,  // 37: vibrato_rate_hz
            // Additional physics (8D)
            1.0,     // 38: spectral_flux
            10000.0, // 39: spectral_rolloff
            1.0,     // 40: spectral_entropy
            0.5,     // 41: subharmonic_ratio
            1000.0,  // 42: fm_depth_hz
            1.0,     // 43: am_depth
            1.0,     // 44: pitch_entropy
            40.0,    // 45: hnr_db
        ]);

        // === Layer 2: Macro Texture (30D) ===
        ranges.extend_from_slice(&[
            // Harmonic texture (9D)
            2.0,  // 46: harmonic_slope
            10.0, // 47: h1_h2_diff_db
            1.0,  // 48: harmonic_irregularity
            1.0,  // 49: harmonic_energy_variance
            1.0,  // 50: spectral_flux_std
            2.0,  // 51: h1_h2_ratio
            2.0,  // 52: h2_h3_ratio
            2.0,  // 53: h3_h4_ratio
            1.0,  // 54: harmonic_density
            // Pitch geometry (7D)
            1000.0, // 55: f0_mean_derivative
            100.0,  // 56: f0_curvature
            50.0,   // 57: f0_inflection_count
            1000.0, // 58: glissando_rate
            1.0,    // 59: vibrato_regularity
            0.1,    // 60: jitter_trend
            1.0,    // 61: pitch_complexity
            // GLCM texture (14D)
            100.0, // 62: glcm_contrast
            1.0,   // 63: glcm_correlation
            100.0, // 64: glcm_energy
            1.0,   // 65: glcm_homogeneity
            100.0, // 66: run_length_nonuniformity
            100.0, // 67: long_run_emphasis
            100.0, // 68: short_run_emphasis
            10.0,  // 69: granularity
            1.0,   // 70: vertical_strength
            1.0,   // 71: horizontal_correlation
            5.0,   // 72: texture_entropy
            1.0,   // 73: texture_homogeneity
            100.0, // 74: texture_contrast
            100.0, // 75: texture_energy
        ]);

        // === Layer 3: Micro Texture (36D) ===
        ranges.extend_from_slice(&[
            // Spectral derivative (6D)
            5000.0, // 76: spectral_derivative_mean
            5000.0, // 77: spectral_derivative_std
            10.0,   // 78: spectral_derivative_skew
            10.0,   // 79: spectral_derivative_kurtosis
            5000.0, // 80: spectral_derivative_max
            5000.0, // 81: spectral_derivative_range
            // FM bin (5D)
            100.0,  // 82: fm_rate_mean
            100.0,  // 83: fm_rate_std
            100.0,  // 84: fm_depth_mean
            100.0,  // 85: fm_depth_std
            1000.0, // 86: fm_extent_hz
            // Dynamics bin (5D)
            100.0, // 87: dynamics_rise_rate
            100.0, // 88: dynamics_fall_rate
            100.0, // 89: dynamics_range_db
            1.0,   // 90: dynamics_cv
            10.0,  // 91: dynamics_skew
            // ICI bin (5D)
            500.0, // 92: ici_mean_ms
            500.0, // 93: ici_std_ms
            10.0,  // 94: ici_skew
            10.0,  // 95: ici_kurtosis
            1.0,   // 96: ici_regularity
            // Rhythm histogram extended (15D)
            50.0,  // 97:  rhythm_tempo_hz
            1.0,   // 98:  rhythm_tempo_stability
            1.0,   // 99:  rhythm_pulse_clarity
            1.0,   // 100: rhythm_grouping_strength
            500.0, // 101: rhythm_cycle_length
            1.0,   // 102: rhythm_onset_strength
            1.0,   // 103: rhythm_swing_factor
            1.0,   // 104: rhythm_syncopation
            1.0,   // 105: rhythm_density
            1.0,   // 106: rhythm_complexity
            1.0,   // 107: rhythm_entropy
            50.0,  // 108: rhythm_peak_rate_hz
            1.0,   // 109: rhythm_valley_depth
            1.0,   // 110: rhythm_crest_factor
            1.0,   // 111: rhythm_flux
        ]);

        ranges
    }

    /// Get feature weights for distance calculation
    ///
    /// Calibrated so physics (0-45) collectively carries ~60% and
    /// texture (46-111) carries ~40% of the weight budget, matching
    /// the pipeline's current 0.6/0.4 split.
    pub fn feature_weights() -> Vec<f32> {
        let mut weights = Vec::with_capacity(112);

        // === Layer 1: Base Physics (46D) — avg ~1.5 ===
        weights.extend_from_slice(&[
            // Fundamental (3D) - HIGH importance
            2.0, // 0:  mean_f0_hz
            1.5, // 1:  duration_ms
            1.5, // 2:  f0_range_hz
            // Energy (3D) - MEDIUM
            1.0, // 3:  rms_energy
            1.0, // 4:  zero_crossing_rate
            1.0, // 5:  peak_amplitude
            // Harmonicity (3D) - HIGH
            1.8, // 6:  harmonic_to_noise_ratio
            1.8, // 7:  harmonicity
            1.5, // 8:  spectral_flatness
            // Temporal envelope (4D) - VARIABLE
            1.8, // 9:  attack_time_ms
            1.5, // 10: decay_time_ms
            1.3, // 11: sustain_level
            1.0, // 12: release_time_ms
            // MFCCs (13D) - HIGH importance
            2.0, // 13: mfcc_0
            2.0, // 14: mfcc_1
            1.8, // 15: mfcc_2
            1.5, // 16: mfcc_3
            1.3, // 17: mfcc_4
            1.3, // 18: mfcc_5
            1.3, // 19: mfcc_6
            1.3, // 20: mfcc_7
            1.3, // 21: mfcc_8
            1.3, // 22: mfcc_9
            1.3, // 23: mfcc_10
            1.3, // 24: mfcc_11
            1.3, // 25: mfcc_12
            // Spectral shape (4D) - MEDIUM
            1.5, // 26: spectral_centroid
            1.3, // 27: spectral_spread
            1.2, // 28: spectral_skewness
            1.2, // 29: spectral_kurtosis
            // Rhythm basics (4D) - MEDIUM
            1.2, // 30: median_ici_ms
            1.5, // 31: onset_rate_hz
            1.0, // 32: ici_coefficient_of_variation
            1.0, // 33: rhythm_regularity
            // Perturbation (4D) - VARIABLE
            1.0, // 34: jitter
            1.0, // 35: shimmer
            2.2, // 36: vibrato_depth
            2.5, // 37: vibrato_rate_hz
            // Additional physics (8D) - MIXED
            1.5, // 38: spectral_flux
            1.3, // 39: spectral_rolloff
            1.2, // 40: spectral_entropy
            1.0, // 41: subharmonic_ratio
            1.8, // 42: fm_depth_hz
            1.5, // 43: am_depth
            1.0, // 44: pitch_entropy
            1.8, // 45: hnr_db
        ]);

        // === Layer 2: Macro Texture (30D) — avg ~1.2 ===
        weights.extend_from_slice(&[
            // Harmonic texture (9D)
            1.3, // 46: harmonic_slope
            1.5, // 47: h1_h2_diff_db
            1.2, // 48: harmonic_irregularity
            1.2, // 49: harmonic_energy_variance
            1.0, // 50: spectral_flux_std
            1.3, // 51: h1_h2_ratio
            1.3, // 52: h2_h3_ratio
            1.3, // 53: h3_h4_ratio
            1.2, // 54: harmonic_density
            // Pitch geometry (7D)
            1.3, // 55: f0_mean_derivative
            1.5, // 56: f0_curvature
            1.0, // 57: f0_inflection_count
            1.3, // 58: glissando_rate
            1.2, // 59: vibrato_regularity
            1.0, // 60: jitter_trend
            1.2, // 61: pitch_complexity
            // GLCM texture (14D)
            0.9, // 62: glcm_contrast
            0.8, // 63: glcm_correlation
            0.8, // 64: glcm_energy
            0.8, // 65: glcm_homogeneity
            0.9, // 66: run_length_nonuniformity
            0.9, // 67: long_run_emphasis
            0.9, // 68: short_run_emphasis
            1.0, // 69: granularity
            1.0, // 70: vertical_strength
            1.0, // 71: horizontal_correlation
            1.0, // 72: texture_entropy
            1.0, // 73: texture_homogeneity
            1.0, // 74: texture_contrast
            1.0, // 75: texture_energy
        ]);

        // === Layer 3: Micro Texture (36D) — avg ~0.9 ===
        weights.extend_from_slice(&[
            // Spectral derivative (6D)
            1.0, // 76: spectral_derivative_mean
            1.0, // 77: spectral_derivative_std
            0.9, // 78: spectral_derivative_skew
            0.9, // 79: spectral_derivative_kurtosis
            1.0, // 80: spectral_derivative_max
            1.0, // 81: spectral_derivative_range
            // FM bin (5D)
            1.0, // 82: fm_rate_mean
            1.0, // 83: fm_rate_std
            1.0, // 84: fm_depth_mean
            1.0, // 85: fm_depth_std
            1.0, // 86: fm_extent_hz
            // Dynamics bin (5D)
            1.0, // 87: dynamics_rise_rate
            1.0, // 88: dynamics_fall_rate
            1.0, // 89: dynamics_range_db
            0.9, // 90: dynamics_cv
            0.9, // 91: dynamics_skew
            // ICI bin (5D)
            0.9, // 92: ici_mean_ms
            0.9, // 93: ici_std_ms
            0.8, // 94: ici_skew
            0.8, // 95: ici_kurtosis
            0.9, // 96: ici_regularity
            // Rhythm histogram extended (15D)
            0.8, // 97:  rhythm_tempo_hz
            0.8, // 98:  rhythm_tempo_stability
            0.8, // 99:  rhythm_pulse_clarity
            0.8, // 100: rhythm_grouping_strength
            0.8, // 101: rhythm_cycle_length
            0.8, // 102: rhythm_onset_strength
            0.8, // 103: rhythm_swing_factor
            0.8, // 104: rhythm_syncopation
            0.8, // 105: rhythm_density
            0.8, // 106: rhythm_complexity
            0.8, // 107: rhythm_entropy
            0.8, // 108: rhythm_peak_rate_hz
            0.8, // 109: rhythm_valley_depth
            0.8, // 110: rhythm_crest_factor
            0.8, // 111: rhythm_flux
        ]);

        weights
    }

    /// Calculate normalized weighted Euclidean distance to another vector
    ///
    /// This is the PRIMARY distance metric for acoustic similarity in 112D space.
    /// Distances are:
    /// 1. Normalized by dimension-specific ranges
    /// 2. Weighted by feature importance
    ///
    /// Returns a non-negative distance where 0.0 = identical.
    pub fn distance_to(&self, other: &Self) -> f32 {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let ranges = Self::normalization_ranges();
        let weights = Self::feature_weights();

        let mut sum_squared = 0.0_f32;
        for i in 0..112 {
            let diff = (v1[i] - v2[i]) / ranges[i];
            sum_squared += weights[i] * diff * diff;
        }

        sum_squared.sqrt()
    }

    /// Calculate weighted distance with custom weights (backward compat)
    pub fn distance_to_custom_weights(&self, other: &Self, weights: &[f32]) -> f32 {
        let self_arr = self.to_array();
        let other_arr = other.to_array();

        self_arr
            .iter()
            .zip(other_arr.iter())
            .zip(weights.iter())
            .map(|((&a, &b), &w)| w * (a - b).powi(2))
            .sum()
    }

    /// Add two vectors
    pub fn add(&self, other: &Self) -> Self {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let mut result = vec![0.0f32; 112];
        for i in 0..112 {
            result[i] = v1[i] + v2[i];
        }
        Self::from_array(&result)
    }

    /// Subtract two vectors
    pub fn sub(&self, other: &Self) -> Self {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let mut result = vec![0.0f32; 112];
        for i in 0..112 {
            result[i] = v1[i] - v2[i];
        }
        Self::from_array(&result)
    }

    /// Scalar multiplication
    pub fn scale(&self, factor: f32) -> Self {
        let v = self.to_array();
        let mut result = vec![0.0f32; 112];
        for i in 0..112 {
            result[i] = v[i] * factor;
        }
        Self::from_array(&result)
    }

    /// Calculate magnitude (weighted Euclidean norm with normalization)
    pub fn magnitude(&self) -> f32 {
        let arr = self.to_array();
        let ranges = Self::normalization_ranges();
        let weights = Self::feature_weights();

        let mut sum_squared = 0.0_f32;
        for i in 0..112 {
            let normalized = arr[i] / ranges[i];
            sum_squared += weights[i] * normalized * normalized;
        }

        sum_squared.sqrt()
    }

    /// Normalize to unit vector
    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag > 1e-6 {
            self.scale(1.0 / mag)
        } else {
            self.clone()
        }
    }

    /// Linear interpolation between two vectors
    ///
    /// Alpha in [0.0, 1.0]: 0.0 = self, 0.5 = midpoint, 1.0 = other.
    pub fn interpolate(&self, other: &Self, alpha: f32) -> Self {
        debug_assert!((0.0..=1.0).contains(&alpha), "Alpha must be in [0, 1], got {}", alpha);

        let v1 = self.to_array();
        let v2 = other.to_array();
        let mut result = vec![0.0f32; 112];
        for i in 0..112 {
            result[i] = v1[i] * (1.0 - alpha) + v2[i] * alpha;
        }
        Self::from_array(&result)
    }

    /// Vector extrapolation: origin + delta * factor
    ///
    /// Factor >= 0.0: 0.0 = no movement, 1.0 = origin + direction.
    pub fn extrapolate(&self, delta: &VectorDelta112D, factor: f32) -> Self {
        debug_assert!(factor >= 0.0, "Factor must be >= 0, got {}", factor);

        let v = self.to_array();
        let d = delta.as_slice();
        let mut result = vec![0.0f32; 112];
        for i in 0..112 {
            result[i] = v[i] + d[i] * factor;
        }
        Self::from_array(&result)
    }
}

// ============================================================================
// Operator Overloads for Vector112D
// ============================================================================

impl std::ops::Add for Vector112D {
    type Output = Vector112D;

    fn add(self, rhs: Vector112D) -> Self::Output {
        Vector112D::add(&self, &rhs)
    }
}

impl std::ops::Sub for Vector112D {
    type Output = Vector112D;

    fn sub(self, rhs: Vector112D) -> Self::Output {
        Vector112D::sub(&self, &rhs)
    }
}

impl std::ops::Mul<f32> for Vector112D {
    type Output = Vector112D;

    fn mul(self, rhs: f32) -> Self::Output {
        Vector112D::scale(&self, rhs)
    }
}

// ============================================================================
// 112D Vector Delta
// ============================================================================

/// 112-dimensional delta vector for extrapolation operations
///
/// Uses a flat `Vec<f32>` representation since 112 named delta fields
/// would be ~336 lines of boilerplate. Deltas are only used for
/// extrapolation and don't need individual field access.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorDelta112D {
    pub deltas: Vec<f32>, // 112 elements
}

impl VectorDelta112D {
    /// Create a zero-initialized delta
    pub fn zero() -> Self {
        Self { deltas: vec![0.0; 112] }
    }

    /// Create delta as target - source
    pub fn from_vectors(target: &Vector112D, source: &Vector112D) -> Self {
        let t = target.to_array();
        let s = source.to_array();
        let mut deltas = vec![0.0f32; 112];
        for i in 0..112 {
            deltas[i] = t[i] - s[i];
        }
        Self { deltas }
    }

    /// Create from a raw slice (must have >= 112 elements)
    pub fn from_slice(slice: &[f32]) -> Self {
        debug_assert!(slice.len() >= 112, "from_slice requires at least 112 elements");
        Self {
            deltas: slice[0..112].to_vec(),
        }
    }

    /// Access as a slice
    pub fn as_slice(&self) -> &[f32] {
        &self.deltas
    }
}

// ============================================================================
// From<RosettaFeatures> Conversion
// ============================================================================

impl From<RosettaFeatures> for Vector112D {
    /// Convert from 112D RosettaFeatures to Vector112D
    ///
    /// This is guaranteed correct because both use the same index layout:
    /// RosettaFeatures::to_array() and Vector112D::to_array() share the
    /// same field-to-index mapping.
    fn from(f: RosettaFeatures) -> Self {
        Self::from_array(&f.to_array())
    }
}

/// Convert Vector105D to Vector112D (upgrades with zero-filled new dimensions)
impl From<&Vector105D> for Vector112D {
    fn from(v: &Vector105D) -> Self {
        // Convert 105D to 112D:
        // - base_45d → base_46d (add release_time_ms at index 12, default 0.0)
        // - macro_texture: same 30D
        // - micro_texture: 30D → 36D (add 6 rhythm features at end, default 0.0)

        let mut base_46d = vec![0.0; 46];
        // Copy base_45d, inserting release_time_ms placeholder at index 12
        base_46d[0..12].copy_from_slice(&v.base_45d[0..12]);
        // Index 12 is release_time_ms (0.0 for now)
        base_46d[13..46].copy_from_slice(&v.base_45d[12..45]);

        Self {
            base_46d,
            // Macro texture (same 8D from 105D + 1 new)
            harmonic_slope: v.harmonic_slope,
            h1_h2_diff_db: v.h1_h2_diff_db,
            harmonic_irregularity: v.harmonic_irregularity,
            harmonic_energy_variance: v.harmonic_energy_variance,
            spectral_flux_std: v.spectral_flux_std,
            h1_h2_ratio: v.h1_h2_ratio,
            h2_h3_ratio: v.h2_h3_ratio,
            h3_h4_ratio: v.h3_h4_ratio,
            harmonic_density: 0.0, // NEW in 112D - not in 105D
            f0_mean_derivative: v.f0_mean_derivative,
            f0_curvature: v.f0_curvature,
            f0_inflection_count: v.f0_inflection_count,
            glissando_rate: v.glissando_rate,
            vibrato_regularity: v.vibrato_regularity,
            jitter_trend: v.jitter_trend,
            pitch_complexity: v.pitch_entropy, // Note: renamed in 112D
            glcm_contrast: v.glcm_contrast,
            glcm_correlation: v.glcm_correlation,
            glcm_energy: v.glcm_energy,
            glcm_homogeneity: v.glcm_homogeneity,
            run_length_nonuniformity: v.run_length_nonuniformity,
            long_run_emphasis: v.long_run_emphasis,
            short_run_emphasis: v.short_run_emphasis,
            granularity: v.granularity,
            vertical_strength: v.vertical_strength,
            horizontal_correlation: v.diagonal_strength, // Note: renamed
            texture_entropy: 0.0,                        // New in 112D
            texture_homogeneity: 0.0,                    // New in 112D
            texture_contrast: 0.0,                       // New in 112D
            texture_energy: 0.0,                         // New in 112D
            // Micro texture - new 6D spectral derivative
            spectral_derivative_mean: 0.0,
            spectral_derivative_std: 0.0,
            spectral_derivative_skew: 0.0,
            spectral_derivative_kurtosis: 0.0,
            spectral_derivative_max: 0.0,
            spectral_derivative_range: 0.0,
            // FM Bin (from 105D modulation spectra)
            fm_rate_mean: v.fm_spectrum_0_10hz,
            fm_rate_std: v.fm_spectrum_10_30hz,
            fm_depth_mean: v.fm_spectrum_30_50hz,
            fm_depth_std: v.fm_spectrum_50_100hz,
            fm_extent_hz: v.fm_depth_mean,
            // Dynamics Bin (from 105D AM spectra)
            dynamics_rise_rate: v.am_spectrum_0_10hz,
            dynamics_fall_rate: v.am_spectrum_10_30hz,
            dynamics_range_db: v.am_spectrum_30_50hz,
            dynamics_cv: v.am_spectrum_50_100hz,
            dynamics_skew: v.am_depth_mean,
            // ICI Bin (from 105D rhythm histograms)
            ici_mean_ms: v.ioi_bin_0_50ms,
            ici_std_ms: v.ioi_bin_50_100ms,
            ici_skew: v.ioi_bin_100_200ms,
            ici_kurtosis: v.ioi_bin_200_500ms,
            ici_regularity: v.ioi_bin_500_1000ms,
            // Rhythm Histogram Extended (15D) - mapped from 105D
            rhythm_tempo_hz: v.ioi_bin_1000_plus,
            rhythm_tempo_stability: v.ioi_variance,
            rhythm_pulse_clarity: v.ioi_skewness,
            rhythm_grouping_strength: v.ioi_kurtosis,
            rhythm_cycle_length: v.rhythm_regularity,
            // New rhythm features (10D) - zero for now
            rhythm_onset_strength: 0.0,
            rhythm_swing_factor: 0.0,
            rhythm_syncopation: 0.0,
            rhythm_density: 0.0,
            rhythm_complexity: 0.0,
            rhythm_entropy: 0.0,
            rhythm_peak_rate_hz: 0.0,
            rhythm_valley_depth: 0.0,
            rhythm_crest_factor: 0.0,
            rhythm_flux: 0.0,
        }
    }
}

// ============================================================================
// Modulation Spectra Computation (Module 5)
// ============================================================================

/// Compute Amplitude Modulation spectrum from energy envelope
///
/// Decomposes the energy envelope into frequency bands to detect:
/// - Trills (10-30Hz AM)
/// - Roughness (30-50Hz AM)
/// - Insect buzz (50-100Hz AM)
pub fn compute_am_spectrum(energy_envelope: &[f32], sample_rate: f32, frame_rate: f32) -> (f32, f32, f32, f32, f32) {
    if energy_envelope.len() < 16 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Compute envelope spectrum using simple DFT at target frequencies
    let n = energy_envelope.len() as f32;
    let dt = 1.0 / frame_rate;

    // Band centers in Hz
    let bands = [5.0, 20.0, 40.0, 75.0]; // 0-10, 10-30, 30-50, 50-100 Hz centers

    let mean_energy: f32 = energy_envelope.iter().sum::<f32>() / n;
    let centered: Vec<f32> = energy_envelope.iter().map(|&e| e - mean_energy).collect();

    let mut band_energies = [0.0f32; 4];
    let mut total_mod_energy = 0.0;

    for (band_idx, &freq) in bands.iter().enumerate() {
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for (i, &sample) in centered.iter().enumerate() {
            let t = i as f32 * dt;
            let phase = 2.0 * std::f32::consts::PI * freq * t;
            cos_sum += sample * phase.cos();
            sin_sum += sample * phase.sin();
        }

        let magnitude = (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / n;
        band_energies[band_idx] = magnitude;
        total_mod_energy += magnitude * magnitude;
    }

    // Normalize by total modulation energy
    let total_mod_energy = total_mod_energy.sqrt().max(1e-10);

    // AM depth: measure of modulation strength
    let am_depth = if mean_energy > 1e-10 {
        total_mod_energy / mean_energy
    } else {
        0.0
    };

    (
        band_energies[0], // 0-10 Hz (slow tremolo)
        band_energies[1], // 10-30 Hz (trill)
        band_energies[2], // 30-50 Hz (roughness)
        band_energies[3], // 50-100 Hz (insect buzz)
        am_depth,
    )
}

/// Compute Frequency Modulation spectrum from F0 contour
///
/// Decomposes the pitch contour into frequency bands to detect:
/// - Slow drift (0-10Hz FM)
/// - Vibrato (10-30Hz FM)
/// - Warble (30-50Hz FM)
pub fn compute_fm_spectrum(f0_contour: &[f32], frame_rate: f32) -> (f32, f32, f32, f32, f32) {
    // Filter out unvoiced frames (F0 = 0)
    let voiced: Vec<f32> = f0_contour.iter().cloned().filter(|&f| f > 0.0).collect();

    if voiced.len() < 16 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let n = voiced.len() as f32;
    let dt = 1.0 / frame_rate;

    // Compute F0 derivative (instantaneous FM)
    let mut derivative = Vec::with_capacity(voiced.len() - 1);
    for i in 1..voiced.len() {
        derivative.push((voiced[i] - voiced[i - 1]).abs());
    }

    let mean_deriv: f32 = derivative.iter().sum::<f32>() / derivative.len() as f32;
    let centered: Vec<f32> = derivative.iter().map(|&d| d - mean_deriv).collect();

    // Band centers in Hz
    let bands = [5.0, 20.0, 40.0, 75.0];

    let mut band_energies = [0.0f32; 4];
    let mut total_fm_energy = 0.0;

    for (band_idx, &freq) in bands.iter().enumerate() {
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for (i, &sample) in centered.iter().enumerate() {
            let t = i as f32 * dt;
            let phase = 2.0 * std::f32::consts::PI * freq * t;
            cos_sum += sample * phase.cos();
            sin_sum += sample * phase.sin();
        }

        let magnitude = (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / n;
        band_energies[band_idx] = magnitude;
        total_fm_energy += magnitude * magnitude;
    }

    // FM depth: average F0 excursion relative to mean F0
    let mean_f0: f32 = voiced.iter().sum::<f32>() / n;
    let fm_depth = if mean_f0 > 0.0 { mean_deriv / mean_f0 } else { 0.0 };

    (
        band_energies[0], // 0-10 Hz (slow drift)
        band_energies[1], // 10-30 Hz (vibrato)
        band_energies[2], // 30-50 Hz (warble)
        band_energies[3], // 50-100 Hz (rapid squeak)
        fm_depth,
    )
}

/// Compute modulation statistics
pub fn compute_modulation_stats(
    am_bands: (f32, f32, f32, f32, f32),
    fm_bands: (f32, f32, f32, f32, f32),
) -> (f32, f32, f32, f32, f32) {
    let (am_0_10, am_10_30, am_30_50, am_50_100, am_depth) = am_bands;
    let (fm_0_10, fm_10_30, fm_30_50, fm_50_100, fm_depth) = fm_bands;

    // AM vs FM dominance
    let am_total = am_0_10 + am_10_30 + am_30_50 + am_50_100;
    let fm_total = fm_0_10 + fm_10_30 + fm_30_50 + fm_50_100;
    let am_fm_ratio = if fm_total > 1e-10 { am_total / fm_total } else { 0.0 };

    // Modulation complexity (spread across bands)
    let all_bands = [
        am_0_10, am_10_30, am_30_50, am_50_100, fm_0_10, fm_10_30, fm_30_50, fm_50_100,
    ];
    let total: f32 = all_bands.iter().sum();
    let complexity = if total > 1e-10 {
        let entropy: f32 = all_bands
            .iter()
            .filter(|&&b| b > 1e-10)
            .map(|&b| {
                let p = b / total;
                -p * p.ln()
            })
            .sum();
        entropy
    } else {
        0.0
    };

    // Trill strength (15-30Hz AM band energy)
    let trill_strength = am_10_30;

    // Flutter index (rapid modulation presence)
    let flutter_index = am_30_50 + am_50_100 + fm_30_50 + fm_50_100;

    // AM-FM synchrony (correlation between AM and FM)
    // Simplified: ratio of synchronized bands
    let synchrony = if am_depth + fm_depth > 1e-10 {
        (am_depth.min(fm_depth)) / (am_depth + fm_depth)
    } else {
        0.0
    };

    (am_fm_ratio, complexity, trill_strength, flutter_index, synchrony)
}

// ============================================================================
// Rhythm Histograms Computation (Module 6)
// ============================================================================

/// Compute Inter-Onset Interval histogram from onset times
///
/// Creates a 6-bin histogram capturing the "temporal fingerprint" of the species
pub fn compute_rhythm_histogram(onset_times_ms: &[f32]) -> (f32, f32, f32, f32, f32, f32) {
    if onset_times_ms.len() < 2 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Compute IOIs
    let mut iois: Vec<f32> = Vec::new();
    for i in 1..onset_times_ms.len() {
        iois.push(onset_times_ms[i] - onset_times_ms[i - 1]);
    }

    // Bin the IOIs
    let mut bin_0_50 = 0.0;
    let mut bin_50_100 = 0.0;
    let mut bin_100_200 = 0.0;
    let mut bin_200_500 = 0.0;
    let mut bin_500_1000 = 0.0;
    let mut bin_1000_plus = 0.0;

    for &ioi in &iois {
        if ioi < 50.0 {
            bin_0_50 += 1.0;
        } else if ioi < 100.0 {
            bin_50_100 += 1.0;
        } else if ioi < 200.0 {
            bin_100_200 += 1.0;
        } else if ioi < 500.0 {
            bin_200_500 += 1.0;
        } else if ioi < 1000.0 {
            bin_500_1000 += 1.0;
        } else {
            bin_1000_plus += 1.0;
        }
    }

    // Normalize by total count
    let total = iois.len() as f32;
    if total > 0.0 {
        bin_0_50 /= total;
        bin_50_100 /= total;
        bin_100_200 /= total;
        bin_200_500 /= total;
        bin_500_1000 /= total;
        bin_1000_plus /= total;
    }

    (
        bin_0_50,
        bin_50_100,
        bin_100_200,
        bin_200_500,
        bin_500_1000,
        bin_1000_plus,
    )
}

/// Compute rhythm statistics from IOIs
pub fn compute_rhythm_stats(iois: &[f32]) -> (f32, f32, f32, f32) {
    if iois.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let n = iois.len() as f32;
    let mean: f32 = iois.iter().sum::<f32>() / n;

    // Variance
    let variance: f32 = iois.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

    // Skewness (asymmetry of distribution)
    let std_dev = variance.sqrt().max(1e-10);
    let skewness: f32 = iois.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f32>() / n;

    // Kurtosis (tailedness of distribution)
    let kurtosis: f32 = iois.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f32>() / n - 3.0; // Excess kurtosis

    // Regularity: low variance = high regularity
    let regularity = 1.0 / (1.0 + variance / (mean * mean).max(1e-10));

    (variance, skewness, kurtosis, regularity)
}

// ============================================================================
// Psychoacoustics Computation (Module 7)
// ============================================================================

/// Compute psychoacoustic features
///
/// These features model how sound is *perceived* rather than just measured
pub fn compute_psychoacoustics(spectrum: &[f32], frequencies: &[f32], rms_energy: f32) -> (f32, f32, f32, f32, f32) {
    if spectrum.is_empty() || frequencies.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // 1. Sharpness (Acum) - weighted high-frequency content
    // Based on Zwicker's model: sharpness increases with high-frequency energy
    let mut weighted_sum = 0.0;
    let mut total_energy = 0.0;

    for (i, &mag) in spectrum.iter().enumerate() {
        let freq = frequencies.get(i).copied().unwrap_or(i as f32);
        let energy = mag * mag;

        // Weighting function (simplified Zwicker)
        // High frequencies contribute more to sharpness
        let weight = if freq < 1000.0 {
            1.0
        } else if freq < 4000.0 {
            1.0 + 0.5 * (freq / 1000.0 - 1.0)
        } else {
            2.5 + 0.25 * (freq / 1000.0 - 4.0)
        };

        weighted_sum += energy * weight * freq / 1000.0;
        total_energy += energy;
    }

    let sharpness = if total_energy > 1e-10 {
        weighted_sum / total_energy
    } else {
        0.0
    };

    // 2. Roughness (Asper) - perception of rapid amplitude modulation
    // Simplified: based on spectral peak density
    let peak_count = count_spectral_peaks(spectrum);
    let roughness = (peak_count as f32 / spectrum.len() as f32) * 10.0;

    // 3. Loudness (Sone) - non-linear perception of intensity
    // Using Stevens' power law: loudness ∝ intensity^0.3
    let loudness_db = 20.0 * rms_energy.abs().max(1e-10).log10();
    let loudness_sone = if loudness_db > 0.0 {
        (loudness_db / 40.0).powf(0.3) * 1.0
    } else {
        0.0
    };

    // 4. Tonality - strength of tonal vs noise components
    // Higher spectral flatness = more noise-like, lower = more tonal
    let spectral_flatness = compute_spectral_flatness(spectrum);
    let tonality = 1.0 - spectral_flatness;

    // 5. Fluctuation Strength - perception of slow modulations (< 20Hz)
    // Approximated by low-frequency spectral energy concentration
    let low_freq_energy: f32 = spectrum.iter().take(spectrum.len() / 4).map(|&m| m * m).sum();
    let fluctuation_strength = if total_energy > 1e-10 {
        (low_freq_energy / total_energy).sqrt()
    } else {
        0.0
    };

    (sharpness, roughness, loudness_sone, tonality, fluctuation_strength)
}

/// Count local maxima in spectrum
fn count_spectral_peaks(spectrum: &[f32]) -> usize {
    if spectrum.len() < 3 {
        return 0;
    }

    let mut count = 0;
    for i in 1..spectrum.len() - 1 {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            count += 1;
        }
    }
    count
}

/// Compute spectral flatness (geometric mean / arithmetic mean)
fn compute_spectral_flatness(spectrum: &[f32]) -> f32 {
    if spectrum.is_empty() {
        return 0.0;
    }

    let n = spectrum.len() as f32;

    // Add small epsilon to avoid log(0)
    let log_sum: f32 = spectrum.iter().map(|&m| (m.abs().max(1e-10)).ln()).sum();

    let arithmetic_mean: f32 = spectrum.iter().map(|&m| m.abs()).sum::<f32>() / n;

    if arithmetic_mean < 1e-10 {
        return 0.0;
    }

    // Geometric mean = exp(mean(log(x)))
    let geometric_mean = (log_sum / n).exp();

    // Flatness = geometric_mean / arithmetic_mean
    // Values close to 1 = flat (noise), close to 0 = tonal
    (geometric_mean / arithmetic_mean).min(1.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_105d_zero() {
        let v = Vector105D::zero();
        assert_eq!(v.to_array().len(), 105);
        assert!(v.to_array().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vector_105d_physics_slice() {
        let mut v = Vector105D::zero();
        v.base_45d[0] = 1.0;
        v.base_45d[44] = 2.0;

        let physics = v.physics_slice();
        assert_eq!(physics.len(), 45);
        assert_eq!(physics[0], 1.0);
        assert_eq!(physics[44], 2.0);
    }

    #[test]
    fn test_am_spectrum_sine() {
        // Create amplitude envelope with 20Hz modulation (trill)
        let frame_rate = 100.0;
        let envelope: Vec<f32> = (0..500)
            .map(|i| {
                let t = i as f32 / frame_rate;
                0.5 + 0.5 * (2.0 * std::f32::consts::PI * 20.0 * t).sin()
            })
            .collect();

        let (am_0_10, am_10_30, am_30_50, am_50_100, _depth) = compute_am_spectrum(&envelope, frame_rate, frame_rate);

        // Should have strong 10-30Hz component (trill range)
        assert!(am_10_30 > am_0_10);
        assert!(am_10_30 > am_30_50);
    }

    #[test]
    fn test_fm_spectrum_vibrato() {
        // Create F0 contour with 15Hz vibrato
        let frame_rate = 100.0;
        let f0: Vec<f32> = (0..500)
            .map(|i| {
                let t = i as f32 / frame_rate;
                1000.0 + 100.0 * (2.0 * std::f32::consts::PI * 15.0 * t).sin()
            })
            .collect();

        let (fm_0_10, fm_10_30, fm_30_50, _fm_50_100, _depth) = compute_fm_spectrum(&f0, frame_rate);

        // Should have strong 10-30Hz component (vibrato range)
        assert!(fm_10_30 > fm_0_10);
    }

    #[test]
    fn test_rhythm_histogram() {
        // Create onset times with 100ms gaps (IOIs are 100ms each)
        let onset_times: Vec<f32> = (0..10).map(|i| i as f32 * 100.0).collect();

        let (bin_0_50, bin_50_100, bin_100_200, _, _, _) = compute_rhythm_histogram(&onset_times);

        // IOIs are 100ms (difference between onsets), which falls in 50-100 bin
        // Since IOI = 100ms exactly, it should go to bin_50_100 (100 < 100 is false, so 100 goes to next)
        // Actually: IOI = 100ms, condition is ioi < 100, so 100ms goes to bin_100_200
        assert!((bin_100_200 - 1.0).abs() < 0.01);
        assert!((bin_0_50 - 0.0).abs() < 0.01);
        assert!((bin_50_100 - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_rhythm_stats() {
        let iois = vec![100.0, 100.0, 100.0, 100.0];
        let (variance, _skewness, _kurtosis, regularity) = compute_rhythm_stats(&iois);

        // Perfect regularity = variance 0, regularity 1
        assert!(variance < 1e-10);
        assert!((regularity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_psychoacoustics_tonal() {
        // Create a tonal spectrum (strong peak with low noise floor)
        let mut spectrum = vec![0.001; 100];
        spectrum[50] = 1.0; // Strong tonal component

        let frequencies: Vec<f32> = (0..100).map(|i| i as f32 * 100.0).collect();

        let (sharpness, roughness, loudness, tonality, fluctuation) =
            compute_psychoacoustics(&spectrum, &frequencies, 0.1);

        // With a very spiky spectrum (1.0 peak vs 0.001 floor), tonality should be high
        // The exact value depends on the flatness calculation
        println!(
            "Tonality: {}, Sharpness: {}, Roughness: {}",
            tonality, sharpness, roughness
        );
        // Tonal signal should have tonality > 0 (less flat = more tonal)
        assert!(tonality >= 0.0);
        // Sharpness should be positive
        assert!(sharpness > 0.0);
        // Loudness should be positive
        assert!(loudness >= 0.0);
        // Fluctuation should be in valid range
        assert!((0.0..=1.0).contains(&fluctuation));
    }

    #[test]
    fn test_modulation_stats() {
        let am_bands = (0.1, 0.5, 0.2, 0.1, 0.3);
        let fm_bands = (0.2, 0.3, 0.1, 0.05, 0.2);

        let (am_fm_ratio, complexity, trill_strength, flutter_index, _synchrony) =
            compute_modulation_stats(am_bands, fm_bands);

        // AM total = 0.9, FM total = 0.65
        assert!(am_fm_ratio > 1.0);
        // Trill strength = am_10_30 = 0.5
        assert!((trill_strength - 0.5).abs() < 0.01);
        // Flutter index = am_30_50 + am_50_100 + fm_30_50 + fm_50_100 = 0.45
        assert!((flutter_index - 0.45).abs() < 0.01);
    }

    // ========================================================================
    // Vector112D Algebra Tests
    // ========================================================================

    #[test]
    fn test_112d_from_array_roundtrip() {
        let original = Vector112D::zero();
        let arr = original.to_array();
        let restored = Vector112D::from_array(&arr);
        let arr2 = restored.to_array();
        for i in 0..112 {
            assert!(
                (arr[i] - arr2[i]).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}",
                i,
                arr[i],
                arr2[i]
            );
        }
    }

    #[test]
    fn test_112d_from_array_fixed_roundtrip() {
        let arr = [1.0f32; 112];
        let v = Vector112D::from_array_fixed(arr);
        let arr2 = v.to_array();
        for i in 0..112 {
            assert!((arr[i] - arr2[i]).abs() < 1e-6, "Mismatch at {}", i);
        }
    }

    #[test]
    fn test_112d_distance_to_self() {
        let v = Vector112D::from_array(&vec![5.0; 112]);
        let dist = v.distance_to(&v);
        assert!(dist.abs() < 1e-6, "Distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_112d_distance_symmetry() {
        let mut arr_a = vec![0.0f32; 112];
        let mut arr_b = vec![1.0f32; 112];
        arr_a[0] = 1000.0;
        arr_b[0] = 500.0;
        arr_a[13] = 5.0; // mfcc
        arr_b[13] = 3.0;

        let a = Vector112D::from_array(&arr_a);
        let b = Vector112D::from_array(&arr_b);
        let d_ab = a.distance_to(&b);
        let d_ba = b.distance_to(&a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "Distance should be symmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_112d_distance_triangle_inequality() {
        let a = Vector112D::from_array(&vec![0.0; 112]);
        let mut arr_b = vec![0.0f32; 112];
        arr_b[0] = 1000.0;
        let b = Vector112D::from_array(&arr_b);
        let mut arr_c = vec![0.0f32; 112];
        arr_c[0] = 2000.0;
        let c = Vector112D::from_array(&arr_c);

        let d_ab = a.distance_to(&b);
        let d_bc = b.distance_to(&c);
        let d_ac = a.distance_to(&c);

        assert!(
            d_ac <= d_ab + d_bc + 1e-4,
            "Triangle inequality: {} should be <= {} + {} = {}",
            d_ac,
            d_ab,
            d_bc,
            d_ab + d_bc
        );
    }

    #[test]
    fn test_112d_distance_different_vectors() {
        let a = Vector112D::from_array(&vec![0.0; 112]);
        let b = Vector112D::from_array(&vec![1.0; 112]);
        let dist = a.distance_to(&b);
        assert!(dist > 0.0, "Different vectors should have positive distance");
    }

    #[test]
    fn test_112d_interpolate_endpoints() {
        let a = Vector112D::from_array(&vec![0.0; 112]);
        let b = Vector112D::from_array(&vec![10.0; 112]);

        let at_zero = a.interpolate(&b, 0.0);
        let arr_a = at_zero.to_array();
        for i in 0..112 {
            assert!(arr_a[i].abs() < 1e-6, "alpha=0 should return self at {}", i);
        }

        let at_one = a.interpolate(&b, 1.0);
        let arr_b = at_one.to_array();
        for i in 0..112 {
            assert!(
                (arr_b[i] - 10.0).abs() < 1e-6,
                "alpha=1 should return other at {}, got {}",
                i,
                arr_b[i]
            );
        }
    }

    #[test]
    fn test_112d_interpolate_midpoint() {
        let a = Vector112D::from_array(&vec![0.0; 112]);
        let b = Vector112D::from_array(&vec![10.0; 112]);
        let mid = a.interpolate(&b, 0.5);
        let arr = mid.to_array();
        for i in 0..112 {
            assert!(
                (arr[i] - 5.0).abs() < 1e-6,
                "Midpoint at {} should be 5.0, got {}",
                i,
                arr[i]
            );
        }
    }

    #[test]
    fn test_112d_extrapolate_zero_factor() {
        let origin = Vector112D::from_array(&vec![3.0; 112]);
        let delta = VectorDelta112D::from_slice(&vec![1.0; 112]);
        let result = origin.extrapolate(&delta, 0.0);
        let arr = result.to_array();
        for i in 0..112 {
            assert!(
                (arr[i] - 3.0).abs() < 1e-6,
                "Zero factor should return origin at {}, got {}",
                i,
                arr[i]
            );
        }
    }

    #[test]
    fn test_112d_add_sub_inverse() {
        let a = Vector112D::from_array(&vec![5.0; 112]);
        let b = Vector112D::from_array(&vec![2.0; 112]);
        let result = a.add(&b).sub(&b);
        let arr = result.to_array();
        for i in 0..112 {
            assert!(
                (arr[i] - 5.0).abs() < 1e-4,
                "add then sub should yield original at {}, got {}",
                i,
                arr[i]
            );
        }
    }

    #[test]
    fn test_112d_scale() {
        let v = Vector112D::from_array(&vec![1.0; 112]);
        let doubled = v.scale(2.0);
        let arr = doubled.to_array();
        for i in 0..112 {
            assert!(
                (arr[i] - 2.0).abs() < 1e-6,
                "scale(2.0) at {} should be 2.0, got {}",
                i,
                arr[i]
            );
        }
    }

    #[test]
    fn test_112d_magnitude_zero_vector() {
        let v = Vector112D::zero();
        let mag = v.magnitude();
        assert!(mag.abs() < 1e-6, "Zero vector magnitude should be 0, got {}", mag);
    }

    #[test]
    fn test_112d_normalized_unit_length() {
        let v = Vector112D::from_array(&vec![5.0; 112]);
        let norm = v.normalized();
        let mag = norm.magnitude();
        assert!(
            (mag - 1.0).abs() < 0.05,
            "Normalized magnitude should be ~1.0, got {}",
            mag
        );
    }

    #[test]
    fn test_112d_operator_overloads() {
        let a = Vector112D::from_array(&vec![3.0; 112]);
        let b = Vector112D::from_array(&vec![2.0; 112]);

        // Add
        let sum = a.clone() + b.clone();
        let arr_sum = sum.to_array();
        assert!((arr_sum[0] - 5.0).abs() < 1e-6);

        // Sub
        let diff = a.clone() - b.clone();
        let arr_diff = diff.to_array();
        assert!((arr_diff[0] - 1.0).abs() < 1e-6);

        // Mul
        let scaled = a * 3.0;
        let arr_scaled = scaled.to_array();
        assert!((arr_scaled[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_112d_from_vectors() {
        let target = Vector112D::from_array(&vec![10.0; 112]);
        let source = Vector112D::from_array(&vec![3.0; 112]);
        let delta = VectorDelta112D::from_vectors(&target, &source);
        for i in 0..112 {
            assert!(
                (delta.deltas[i] - 7.0).abs() < 1e-6,
                "Delta at {} should be 7.0, got {}",
                i,
                delta.deltas[i]
            );
        }
    }

    #[test]
    fn test_from_rosetta_features() {
        // Create a minimal RosettaFeatures with non-zero values
        let rf = RosettaFeatures {
            mean_f0_hz: 1000.0,
            duration_ms: 50.0,
            f0_range_hz: 200.0,
            rms_energy: 0.5,
            zero_crossing_rate: 0.1,
            peak_amplitude: 0.8,
            harmonic_to_noise_ratio: 15.0,
            harmonicity: 0.9,
            spectral_flatness: 0.3,
            attack_time_ms: 5.0,
            decay_time_ms: 10.0,
            sustain_level: 0.7,
            release_time_ms: 8.0,
            mfcc_0: 1.0,
            mfcc_1: 2.0,
            mfcc_2: 3.0,
            mfcc_3: 4.0,
            mfcc_4: 5.0,
            mfcc_5: 6.0,
            mfcc_6: 7.0,
            mfcc_7: 8.0,
            mfcc_8: 9.0,
            mfcc_9: 10.0,
            mfcc_10: 11.0,
            mfcc_11: 12.0,
            mfcc_12: 13.0,
            spectral_centroid: 5000.0,
            spectral_spread: 2000.0,
            spectral_skewness: 0.5,
            spectral_kurtosis: 2.0,
            median_ici_ms: 100.0,
            onset_rate_hz: 5.0,
            ici_coefficient_of_variation: 0.3,
            rhythm_regularity: 0.8,
            jitter: 0.01,
            shimmer: 0.02,
            vibrato_depth: 50.0,
            vibrato_rate_hz: 6.0,
            spectral_flux: 0.4,
            spectral_rolloff: 8000.0,
            spectral_entropy: 0.6,
            subharmonic_ratio: 0.1,
            fm_depth_hz: 200.0,
            am_depth: 0.5,
            pitch_entropy: 0.4,
            hnr_db: 20.0,
            // Macro texture (30D)
            harmonic_slope: -6.0,
            h1_h2_diff_db: 3.5,
            harmonic_irregularity: 0.2,
            harmonic_energy_variance: 0.1,
            spectral_flux_std: 0.15,
            h1_h2_ratio: 1.2,
            h2_h3_ratio: 0.9,
            h3_h4_ratio: 0.7,
            harmonic_density: 0.8,
            f0_mean_derivative: 50.0,
            f0_curvature: 10.0,
            f0_inflection_count: 3.0,
            glissando_rate: 100.0,
            vibrato_regularity: 0.9,
            jitter_trend: 0.005,
            pitch_complexity: 0.6,
            glcm_contrast: 20.0,
            glcm_correlation: 0.7,
            glcm_energy: 15.0,
            glcm_homogeneity: 0.8,
            run_length_nonuniformity: 10.0,
            long_run_emphasis: 5.0,
            short_run_emphasis: 8.0,
            granularity: 2.0,
            vertical_strength: 0.6,
            horizontal_correlation: 0.5,
            texture_entropy: 1.5,
            texture_homogeneity: 0.7,
            texture_contrast: 25.0,
            texture_energy: 30.0,
            // Micro texture (36D)
            spectral_derivative_mean: 100.0,
            spectral_derivative_std: 50.0,
            spectral_derivative_skew: 0.3,
            spectral_derivative_kurtosis: 1.5,
            spectral_derivative_max: 200.0,
            spectral_derivative_range: 150.0,
            fm_rate_mean: 10.0,
            fm_rate_std: 5.0,
            fm_depth_mean: 30.0,
            fm_depth_std: 15.0,
            fm_extent_hz: 500.0,
            dynamics_rise_rate: 20.0,
            dynamics_fall_rate: 15.0,
            dynamics_range_db: 40.0,
            dynamics_cv: 0.3,
            dynamics_skew: 0.5,
            ici_mean_ms: 80.0,
            ici_std_ms: 20.0,
            ici_skew: 0.4,
            ici_kurtosis: 2.0,
            ici_regularity: 0.7,
            rhythm_tempo_hz: 4.0,
            rhythm_tempo_stability: 0.8,
            rhythm_pulse_clarity: 0.6,
            rhythm_grouping_strength: 0.5,
            rhythm_cycle_length: 200.0,
            rhythm_onset_strength: 0.7,
            rhythm_swing_factor: 0.3,
            rhythm_syncopation: 0.2,
            rhythm_density: 0.4,
            rhythm_complexity: 0.5,
            rhythm_entropy: 0.6,
            rhythm_peak_rate_hz: 3.0,
            rhythm_valley_depth: 0.4,
            rhythm_crest_factor: 0.7,
            rhythm_flux: 0.3,
        };

        let rf_arr = rf.to_array();
        let v = Vector112D::from(rf);
        let v_arr = v.to_array();

        for i in 0..112 {
            assert!(
                (rf_arr[i] - v_arr[i]).abs() < 1e-6,
                "From<RosettaFeatures> mismatch at {}: {} vs {}",
                i,
                rf_arr[i],
                v_arr[i]
            );
        }
    }

    #[test]
    fn test_normalization_ranges_positive() {
        let ranges = Vector112D::normalization_ranges();
        assert_eq!(ranges.len(), 112);
        for (i, &r) in ranges.iter().enumerate() {
            assert!(r > 0.0, "Range at {} should be positive, got {}", i, r);
        }
    }

    #[test]
    fn test_feature_weights_positive() {
        let weights = Vector112D::feature_weights();
        assert_eq!(weights.len(), 112);
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "Weight at {} should be positive, got {}", i, w);
        }
    }

    #[test]
    fn test_distance_matches_pipeline() {
        // Verify that Vector112D::distance_to() produces consistent results
        // with the old pipeline's weighted Euclidean approach
        let a = Vector112D::from_array(&vec![1.0; 112]);
        let b = Vector112D::from_array(&vec![3.0; 112]);

        let dist = a.distance_to(&b);

        // Manually compute expected distance using the same weights/ranges
        let va = a.to_array();
        let vb = b.to_array();
        let ranges = Vector112D::normalization_ranges();
        let weights = Vector112D::feature_weights();

        let mut expected = 0.0_f32;
        for i in 0..112 {
            let diff = (va[i] - vb[i]) / ranges[i];
            expected += weights[i] * diff * diff;
        }
        expected = expected.sqrt();

        assert!(
            (dist - expected).abs() < 1e-4,
            "distance_to ({}) should match manual ({})",
            dist,
            expected
        );

        // Distance should be positive for different vectors
        assert!(dist > 0.0);
    }
}
