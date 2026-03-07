//! Acoustic Algebra 45D Module (Rust Execution Layer)
//! ===================================================
//!
//! This module implements 45-dimensional acoustic vector operations for
//! high-fidelity synthesis and acoustic analysis. It extends the 30D
//! Island Hopping vectors with 15 additional bioacoustic features.
//!
//! **Feature Layout (45D):**
//! - Fundamental (3): mean_f0_hz, duration_ms, f0_range_hz
//! - Grit (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
//! - Motion (7): attack, decay, sustain, vibrato_rate, vibrato_depth, jitter, shimmer
//! - Fingerprint (14): mfcc_1-13, spectral_flux
//! - Rhythm (3): median_ici, onset_rate, ici_cv
//! - Resonance (6): formant_1-3, bandwidth_1-2, dispersion
//! - Spectral Shape (4): centroid, spread, skewness, kurtosis
//! - Modulation (3): spectral_tilt, fm_slope, am_depth
//! - Non-Linear (2): subharmonic_ratio, spectral_entropy
//!
//! **Key Operations:**
//! - `distance_to()`: Weighted Euclidean distance for similarity
//! - `interpolate()`: Linear interpolation between vectors (Bridge Builder)
//! - `extrapolate()`: Apply delta with factor (Ocean Explorer)
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::micro_dynamics_extractor::MicroDynamicsFeatures45D;
use serde::{Deserialize, Serialize};

// ============================================================================
// 45D Vector Definition
// ============================================================================

/// 45-dimensional acoustic feature vector for synthesis and analysis
///
/// Features organized by category following MicroDynamicsFeatures45D layout:
/// - Fundamental (3): Core pitch and timing
/// - Grit (3): Harmonic quality and noise
/// - Motion (7): Envelope and perturbation
/// - Fingerprint (14): Spectral envelope
/// - Rhythm (3): Temporal patterns
/// - Resonance (6): Formant structure
/// - Spectral Shape (4): Spectral moments
/// - Modulation (3): FM/AM characteristics
/// - Non-Linear (2): Complex dynamics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector45D {
    // === Fundamental (3) ===
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,

    // === Grit Factors (3) ===
    pub harmonic_to_noise_ratio: f32,
    pub spectral_flatness: f32,
    pub harmonicity: f32,

    // === Motion Factors (7) ===
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    pub sustain_level: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth: f32,
    pub jitter: f32,
    pub shimmer: f32,

    // === Fingerprint Factors (14) ===
    pub mfcc_1: f32,
    pub mfcc_2: f32,
    pub mfcc_3: f32,
    pub mfcc_4: f32,
    pub mfcc_5: f32,
    pub mfcc_6: f32,
    pub mfcc_7: f32,
    pub mfcc_8: f32,
    pub mfcc_9: f32,
    pub mfcc_10: f32,
    pub mfcc_11: f32,
    pub mfcc_12: f32,
    pub mfcc_13: f32,
    pub spectral_flux: f32,

    // === Rhythm Factors (3) ===
    pub median_ici_ms: f32,
    pub onset_rate_hz: f32,
    pub ici_coefficient_of_variation: f32,

    // === Resonance Factors (6) ===
    pub formant_1_hz: f32,
    pub formant_2_hz: f32,
    pub formant_3_hz: f32,
    pub formant_1_bandwidth: f32,
    pub formant_2_bandwidth: f32,
    pub formant_dispersion: f32,

    // === Spectral Shape Factors (4) ===
    pub spectral_centroid: f32,
    pub spectral_spread: f32,
    pub spectral_skewness: f32,
    pub spectral_kurtosis: f32,

    // === Modulation Factors (3) ===
    pub spectral_tilt: f32,
    pub fm_slope: f32,
    pub am_depth: f32,

    // === Non-Linear Factors (2) ===
    pub subharmonic_ratio: f32,
    pub spectral_entropy: f32,
}

impl Default for Vector45D {
    fn default() -> Self {
        Self {
            // Fundamental (3) - bioacoustic defaults
            mean_f0_hz: 7000.0,
            duration_ms: 50.0,
            f0_range_hz: 400.0,
            // Grit (3)
            harmonic_to_noise_ratio: 20.0,
            spectral_flatness: 0.3,
            harmonicity: 0.8,
            // Motion (7)
            attack_time_ms: 5.0,
            decay_time_ms: 20.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 7.0,
            vibrato_depth: 50.0,
            jitter: 0.01,
            shimmer: 0.03,
            // Fingerprint (14) - MFCCs decay
            mfcc_1: -10.0,
            mfcc_2: -5.0,
            mfcc_3: -2.0,
            mfcc_4: -1.0,
            mfcc_5: -0.5,
            mfcc_6: -0.3,
            mfcc_7: -0.2,
            mfcc_8: -0.1,
            mfcc_9: 0.0,
            mfcc_10: 0.1,
            mfcc_11: 0.2,
            mfcc_12: 0.3,
            mfcc_13: 0.4,
            spectral_flux: 0.5,
            // Rhythm (3)
            median_ici_ms: 15.0,
            onset_rate_hz: 8.0,
            ici_coefficient_of_variation: 0.3,
            // Resonance (6) - typical marmoset formants
            formant_1_hz: 1500.0,
            formant_2_hz: 3000.0,
            formant_3_hz: 4500.0,
            formant_1_bandwidth: 100.0,
            formant_2_bandwidth: 150.0,
            formant_dispersion: 1500.0,
            // Spectral Shape (4)
            spectral_centroid: 5000.0,
            spectral_spread: 2000.0,
            spectral_skewness: 0.0,
            spectral_kurtosis: 3.0,
            // Modulation (3)
            spectral_tilt: -6.0,
            fm_slope: 0.0,
            am_depth: 0.5,
            // Non-Linear (2)
            subharmonic_ratio: 0.0,
            spectral_entropy: 0.3,
        }
    }
}

impl Vector45D {
    /// Convert to flat 45D array for ML use
    ///
    /// Layout matches MicroDynamicsFeatures45D::to_array():
    /// - [0-2]: Fundamental (mean_f0_hz, duration_ms, f0_range_hz)
    /// - [3-5]: Grit Factors (hnr, spectral_flatness, harmonicity)
    /// - [6-12]: Motion Factors (attack, decay, sustain, vibrato_rate, vibrato_depth, jitter, shimmer)
    /// - [13-26]: Fingerprint (mfcc_1-13, spectral_flux)
    /// - [27-29]: Rhythm (median_ici, onset_rate, ici_cv)
    /// - [30-35]: Resonance (formant_1-3, bandwidth_1-2, dispersion)
    /// - [36-39]: Spectral Shape (centroid, spread, skewness, kurtosis)
    /// - [40-42]: Modulation (spectral_tilt, fm_slope, am_depth)
    /// - [43-44]: Non-Linear (subharmonic_ratio, spectral_entropy)
    pub fn to_array(&self) -> [f32; 45] {
        let mut arr = [0.0f32; 45];

        // Fundamental (3)
        arr[0] = self.mean_f0_hz;
        arr[1] = self.duration_ms;
        arr[2] = self.f0_range_hz;

        // Grit Factors (3)
        arr[3] = self.harmonic_to_noise_ratio;
        arr[4] = self.spectral_flatness;
        arr[5] = self.harmonicity;

        // Motion Factors (7)
        arr[6] = self.attack_time_ms;
        arr[7] = self.decay_time_ms;
        arr[8] = self.sustain_level;
        arr[9] = self.vibrato_rate_hz;
        arr[10] = self.vibrato_depth;
        arr[11] = self.jitter;
        arr[12] = self.shimmer;

        // Fingerprint (14) - MFCCs 1-13 + spectral_flux
        arr[13] = self.mfcc_1;
        arr[14] = self.mfcc_2;
        arr[15] = self.mfcc_3;
        arr[16] = self.mfcc_4;
        arr[17] = self.mfcc_5;
        arr[18] = self.mfcc_6;
        arr[19] = self.mfcc_7;
        arr[20] = self.mfcc_8;
        arr[21] = self.mfcc_9;
        arr[22] = self.mfcc_10;
        arr[23] = self.mfcc_11;
        arr[24] = self.mfcc_12;
        arr[25] = self.mfcc_13;
        arr[26] = self.spectral_flux;

        // Rhythm (3)
        arr[27] = self.median_ici_ms;
        arr[28] = self.onset_rate_hz;
        arr[29] = self.ici_coefficient_of_variation;

        // Resonance (6)
        arr[30] = self.formant_1_hz;
        arr[31] = self.formant_2_hz;
        arr[32] = self.formant_3_hz;
        arr[33] = self.formant_1_bandwidth;
        arr[34] = self.formant_2_bandwidth;
        arr[35] = self.formant_dispersion;

        // Spectral Shape (4)
        arr[36] = self.spectral_centroid;
        arr[37] = self.spectral_spread;
        arr[38] = self.spectral_skewness;
        arr[39] = self.spectral_kurtosis;

        // Modulation (3)
        arr[40] = self.spectral_tilt;
        arr[41] = self.fm_slope;
        arr[42] = self.am_depth;

        // Non-Linear (2)
        arr[43] = self.subharmonic_ratio;
        arr[44] = self.spectral_entropy;

        arr
    }

    /// Convert from flat 45D array
    pub fn from_array(arr: [f32; 45]) -> Self {
        Self {
            // Fundamental (3)
            mean_f0_hz: arr[0],
            duration_ms: arr[1],
            f0_range_hz: arr[2],
            // Grit (3)
            harmonic_to_noise_ratio: arr[3],
            spectral_flatness: arr[4],
            harmonicity: arr[5],
            // Motion (7)
            attack_time_ms: arr[6],
            decay_time_ms: arr[7],
            sustain_level: arr[8],
            vibrato_rate_hz: arr[9],
            vibrato_depth: arr[10],
            jitter: arr[11],
            shimmer: arr[12],
            // Fingerprint (14)
            mfcc_1: arr[13],
            mfcc_2: arr[14],
            mfcc_3: arr[15],
            mfcc_4: arr[16],
            mfcc_5: arr[17],
            mfcc_6: arr[18],
            mfcc_7: arr[19],
            mfcc_8: arr[20],
            mfcc_9: arr[21],
            mfcc_10: arr[22],
            mfcc_11: arr[23],
            mfcc_12: arr[24],
            mfcc_13: arr[25],
            spectral_flux: arr[26],
            // Rhythm (3)
            median_ici_ms: arr[27],
            onset_rate_hz: arr[28],
            ici_coefficient_of_variation: arr[29],
            // Resonance (6)
            formant_1_hz: arr[30],
            formant_2_hz: arr[31],
            formant_3_hz: arr[32],
            formant_1_bandwidth: arr[33],
            formant_2_bandwidth: arr[34],
            formant_dispersion: arr[35],
            // Spectral Shape (4)
            spectral_centroid: arr[36],
            spectral_spread: arr[37],
            spectral_skewness: arr[38],
            spectral_kurtosis: arr[39],
            // Modulation (3)
            spectral_tilt: arr[40],
            fm_slope: arr[41],
            am_depth: arr[42],
            // Non-Linear (2)
            subharmonic_ratio: arr[43],
            spectral_entropy: arr[44],
        }
    }

    /// Get normalization ranges for each dimension
    ///
    /// These ranges are used to normalize features for distance calculation,
    /// ensuring meaningful comparisons across different acoustic dimensions.
    pub fn normalization_ranges() -> [f32; 45] {
        [
            // Fundamental (3)
            2000.0, // mean_f0_hz
            100.0,  // duration_ms
            500.0,  // f0_range_hz
            // Grit (3)
            30.0, // harmonic_to_noise_ratio
            1.0,  // spectral_flatness
            1.0,  // harmonicity
            // Motion (7)
            20.0,  // attack_time_ms
            50.0,  // decay_time_ms
            1.0,   // sustain_level
            20.0,  // vibrato_rate_hz
            100.0, // vibrato_depth (cents or Hz equivalent)
            0.05,  // jitter
            0.1,   // shimmer
            // Fingerprint (14)
            20.0, // mfcc_1
            20.0, // mfcc_2
            20.0, // mfcc_3
            20.0, // mfcc_4
            20.0, // mfcc_5
            20.0, // mfcc_6
            20.0, // mfcc_7
            20.0, // mfcc_8
            20.0, // mfcc_9
            20.0, // mfcc_10
            20.0, // mfcc_11
            20.0, // mfcc_12
            20.0, // mfcc_13
            1.0,  // spectral_flux
            // Rhythm (3)
            200.0, // median_ici_ms
            20.0,  // onset_rate_hz
            1.0,   // ici_coefficient_of_variation
            // Resonance (6)
            5000.0,  // formant_1_hz
            8000.0,  // formant_2_hz
            12000.0, // formant_3_hz
            500.0,   // formant_1_bandwidth
            500.0,   // formant_2_bandwidth
            3000.0,  // formant_dispersion
            // Spectral Shape (4)
            15000.0, // spectral_centroid
            5000.0,  // spectral_spread
            2.0,     // spectral_skewness
            5.0,     // spectral_kurtosis
            // Modulation (3)
            12.0,  // spectral_tilt (dB/octave, typically -12 to 0)
            100.0, // fm_slope
            1.0,   // am_depth
            // Non-Linear (2)
            0.5, // subharmonic_ratio
            1.0, // spectral_entropy
        ]
    }

    /// Get feature weights for distance calculation
    ///
    /// These weights reflect the relative importance of each feature group
    /// for bioacoustic similarity assessment.
    pub fn feature_weights() -> [f32; 45] {
        [
            // Fundamental (3) - HIGH importance
            2.0, // mean_f0_hz
            1.5, // duration_ms
            1.5, // f0_range_hz
            // Grit (3) - HIGH importance
            1.8, // harmonic_to_noise_ratio
            1.5, // spectral_flatness
            1.8, // harmonicity
            // Motion (7) - VARIABLE importance
            1.8, // attack_time_ms
            1.5, // decay_time_ms
            1.3, // sustain_level
            2.5, // vibrato_rate_hz - CRITICAL for trill identification
            2.2, // vibrato_depth
            1.0, // jitter
            1.0, // shimmer
            // Fingerprint (14) - HIGH importance
            2.0, // mfcc_1 - energy/brightness
            1.8, // mfcc_2 - spectral shape
            1.5, // mfcc_3
            1.3, // mfcc_4
            1.3, // mfcc_5
            1.3, // mfcc_6
            1.3, // mfcc_7
            1.3, // mfcc_8
            1.3, // mfcc_9
            1.3, // mfcc_10
            1.3, // mfcc_11
            1.3, // mfcc_12
            1.3, // mfcc_13
            1.5, // spectral_flux
            // Rhythm (3) - MEDIUM importance
            1.2, // median_ici_ms
            1.5, // onset_rate_hz
            1.0, // ici_coefficient_of_variation
            // Resonance (6) - HIGH for timbre
            1.8, // formant_1_hz
            1.6, // formant_2_hz
            1.4, // formant_3_hz
            1.2, // formant_1_bandwidth
            1.2, // formant_2_bandwidth
            1.5, // formant_dispersion
            // Spectral Shape (4) - MEDIUM for brightness
            1.5, // spectral_centroid
            1.3, // spectral_spread
            1.2, // spectral_skewness
            1.2, // spectral_kurtosis
            // Modulation (3) - HIGH for dynamics
            1.5, // spectral_tilt
            1.8, // fm_slope
            1.5, // am_depth
            // Non-Linear (2) - MEDIUM for complexity
            1.0, // subharmonic_ratio
            1.2, // spectral_entropy
        ]
    }

    /// Calculate weighted Euclidean distance to another vector
    ///
    /// This is the PRIMARY distance metric for acoustic similarity in 45D space.
    /// Distances are:
    /// 1. Normalized by dimension-specific ranges
    /// 2. Weighted by feature importance
    ///
    /// Returns a non-negative distance where 0.0 = identical.
    pub fn distance_to(&self, other: &Vector45D) -> f32 {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let ranges = Self::normalization_ranges();
        let weights = Self::feature_weights();

        let mut sum_squared = 0.0_f32;
        for i in 0..45 {
            let diff = (v1[i] - v2[i]) / ranges[i];
            sum_squared += weights[i] * diff * diff;
        }

        sum_squared.sqrt()
    }

    /// Linear interpolation between two vectors (Bridge Builder)
    ///
    /// This is SAFE navigation between two known acoustic points.
    /// Alpha must be in [0.0, 1.0]:
    /// - 0.0 = return self
    /// - 0.5 = midpoint
    /// - 1.0 = return other
    pub fn interpolate(&self, other: &Vector45D, alpha: f32) -> Vector45D {
        debug_assert!((0.0..=1.0).contains(&alpha), "Alpha must be in [0, 1], got {}", alpha);

        let v1 = self.to_array();
        let v2 = other.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v1[i] * (1.0 - alpha) + v2[i] * alpha;
        }

        Self::from_array(result)
    }

    /// Vector extrapolation beyond origin (Ocean Explorer)
    ///
    /// This is RISKY navigation beyond known acoustic points into "open ocean".
    /// Factor must be >= 0.0:
    /// - 0.0 = return origin (no movement)
    /// - 1.0 = move to origin + direction
    /// - 2.0 = move twice as far in direction
    pub fn extrapolate(&self, delta: &VectorDelta45D, factor: f32) -> Vector45D {
        debug_assert!(factor >= 0.0, "Factor must be >= 0, got {}", factor);

        let v = self.to_array();
        let d = delta.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v[i] + d[i] * factor;
        }

        Self::from_array(result)
    }

    /// Add two vectors
    pub fn add(&self, other: &Vector45D) -> Vector45D {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v1[i] + v2[i];
        }

        Self::from_array(result)
    }

    /// Subtract two vectors (for delta calculation)
    pub fn sub(&self, other: &Vector45D) -> Vector45D {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v1[i] - v2[i];
        }

        Self::from_array(result)
    }

    /// Scalar multiplication
    pub fn scale(&self, factor: f32) -> Vector45D {
        let v = self.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v[i] * factor;
        }

        Self::from_array(result)
    }

    /// Calculate magnitude (weighted Euclidean norm)
    pub fn magnitude(&self) -> f32 {
        let arr = self.to_array();
        let ranges = Self::normalization_ranges();
        let weights = Self::feature_weights();

        let mut sum_squared = 0.0_f32;
        for i in 0..45 {
            let normalized = arr[i] / ranges[i];
            sum_squared += weights[i] * normalized * normalized;
        }

        sum_squared.sqrt()
    }

    /// Normalize to unit vector
    pub fn normalized(&self) -> Vector45D {
        let mag = self.magnitude();
        if mag > 1e-6 {
            self.scale(1.0 / mag)
        } else {
            *self
        }
    }
}

impl From<MicroDynamicsFeatures45D> for Vector45D {
    fn from(features: MicroDynamicsFeatures45D) -> Self {
        Self {
            // Fundamental (3)
            mean_f0_hz: features.mean_f0_hz,
            duration_ms: features.duration_ms,
            f0_range_hz: features.f0_range_hz,
            // Grit (3)
            harmonic_to_noise_ratio: features.base_30d.harmonic_to_noise_ratio,
            spectral_flatness: features.base_30d.spectral_flatness,
            harmonicity: features.base_30d.harmonicity,
            // Motion (7)
            attack_time_ms: features.base_30d.attack_time_ms,
            decay_time_ms: features.base_30d.decay_time_ms,
            sustain_level: features.base_30d.sustain_level,
            vibrato_rate_hz: features.base_30d.vibrato_rate_hz,
            vibrato_depth: features.base_30d.vibrato_depth,
            jitter: features.base_30d.jitter,
            shimmer: features.base_30d.shimmer,
            // Fingerprint (14)
            mfcc_1: features.base_30d.mfcc[0],
            mfcc_2: features.base_30d.mfcc[1],
            mfcc_3: features.base_30d.mfcc[2],
            mfcc_4: features.base_30d.mfcc[3],
            mfcc_5: features.base_30d.mfcc[4],
            mfcc_6: features.base_30d.mfcc[5],
            mfcc_7: features.base_30d.mfcc[6],
            mfcc_8: features.base_30d.mfcc[7],
            mfcc_9: features.base_30d.mfcc[8],
            mfcc_10: features.base_30d.mfcc[9],
            mfcc_11: features.base_30d.mfcc[10],
            mfcc_12: features.base_30d.mfcc[11],
            mfcc_13: features.base_30d.mfcc[12],
            spectral_flux: features.base_30d.spectral_flux,
            // Rhythm (3)
            median_ici_ms: features.base_30d.median_ici_ms,
            onset_rate_hz: features.base_30d.onset_rate_hz,
            ici_coefficient_of_variation: features.base_30d.ici_coefficient_of_variation,
            // Resonance (6)
            formant_1_hz: features.formant_1_hz,
            formant_2_hz: features.formant_2_hz,
            formant_3_hz: features.formant_3_hz,
            formant_1_bandwidth: features.formant_1_bandwidth,
            formant_2_bandwidth: features.formant_2_bandwidth,
            formant_dispersion: features.formant_dispersion,
            // Spectral Shape (4)
            spectral_centroid: features.spectral_centroid,
            spectral_spread: features.spectral_spread,
            spectral_skewness: features.spectral_skewness,
            spectral_kurtosis: features.spectral_kurtosis,
            // Modulation (3)
            spectral_tilt: features.spectral_tilt,
            fm_slope: features.fm_slope,
            am_depth: features.am_depth,
            // Non-Linear (2)
            subharmonic_ratio: features.subharmonic_ratio,
            spectral_entropy: features.spectral_entropy,
        }
    }
}

impl std::ops::Add<Vector45D> for Vector45D {
    type Output = Vector45D;

    fn add(self, rhs: Vector45D) -> Self::Output {
        let v1 = self.to_array();
        let v2 = rhs.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v1[i] + v2[i];
        }

        Self::from_array(result)
    }
}

impl std::ops::Sub<Vector45D> for Vector45D {
    type Output = Vector45D;

    fn sub(self, rhs: Vector45D) -> Self::Output {
        let v1 = self.to_array();
        let v2 = rhs.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = v1[i] - v2[i];
        }

        Self::from_array(result)
    }
}

impl std::ops::Mul<f32> for Vector45D {
    type Output = Vector45D;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

// ============================================================================
// 45D Vector Delta
// ============================================================================

/// 45-dimensional delta vector for extrapolation operations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VectorDelta45D {
    // === Fundamental (3) ===
    pub delta_mean_f0_hz: f32,
    pub delta_duration_ms: f32,
    pub delta_f0_range_hz: f32,

    // === Grit (3) ===
    pub delta_harmonic_to_noise_ratio: f32,
    pub delta_spectral_flatness: f32,
    pub delta_harmonicity: f32,

    // === Motion (7) ===
    pub delta_attack_time_ms: f32,
    pub delta_decay_time_ms: f32,
    pub delta_sustain_level: f32,
    pub delta_vibrato_rate_hz: f32,
    pub delta_vibrato_depth: f32,
    pub delta_jitter: f32,
    pub delta_shimmer: f32,

    // === Fingerprint (14) ===
    pub delta_mfcc_1: f32,
    pub delta_mfcc_2: f32,
    pub delta_mfcc_3: f32,
    pub delta_mfcc_4: f32,
    pub delta_mfcc_5: f32,
    pub delta_mfcc_6: f32,
    pub delta_mfcc_7: f32,
    pub delta_mfcc_8: f32,
    pub delta_mfcc_9: f32,
    pub delta_mfcc_10: f32,
    pub delta_mfcc_11: f32,
    pub delta_mfcc_12: f32,
    pub delta_mfcc_13: f32,
    pub delta_spectral_flux: f32,

    // === Rhythm (3) ===
    pub delta_median_ici_ms: f32,
    pub delta_onset_rate_hz: f32,
    pub delta_ici_coefficient_of_variation: f32,

    // === Resonance (6) ===
    pub delta_formant_1_hz: f32,
    pub delta_formant_2_hz: f32,
    pub delta_formant_3_hz: f32,
    pub delta_formant_1_bandwidth: f32,
    pub delta_formant_2_bandwidth: f32,
    pub delta_formant_dispersion: f32,

    // === Spectral Shape (4) ===
    pub delta_spectral_centroid: f32,
    pub delta_spectral_spread: f32,
    pub delta_spectral_skewness: f32,
    pub delta_spectral_kurtosis: f32,

    // === Modulation (3) ===
    pub delta_spectral_tilt: f32,
    pub delta_fm_slope: f32,
    pub delta_am_depth: f32,

    // === Non-Linear (2) ===
    pub delta_subharmonic_ratio: f32,
    pub delta_spectral_entropy: f32,
}

impl VectorDelta45D {
    /// Create a zero delta (no change)
    pub fn zero() -> Self {
        Self {
            delta_mean_f0_hz: 0.0,
            delta_duration_ms: 0.0,
            delta_f0_range_hz: 0.0,
            delta_harmonic_to_noise_ratio: 0.0,
            delta_spectral_flatness: 0.0,
            delta_harmonicity: 0.0,
            delta_attack_time_ms: 0.0,
            delta_decay_time_ms: 0.0,
            delta_sustain_level: 0.0,
            delta_vibrato_rate_hz: 0.0,
            delta_vibrato_depth: 0.0,
            delta_jitter: 0.0,
            delta_shimmer: 0.0,
            delta_mfcc_1: 0.0,
            delta_mfcc_2: 0.0,
            delta_mfcc_3: 0.0,
            delta_mfcc_4: 0.0,
            delta_mfcc_5: 0.0,
            delta_mfcc_6: 0.0,
            delta_mfcc_7: 0.0,
            delta_mfcc_8: 0.0,
            delta_mfcc_9: 0.0,
            delta_mfcc_10: 0.0,
            delta_mfcc_11: 0.0,
            delta_mfcc_12: 0.0,
            delta_mfcc_13: 0.0,
            delta_spectral_flux: 0.0,
            delta_median_ici_ms: 0.0,
            delta_onset_rate_hz: 0.0,
            delta_ici_coefficient_of_variation: 0.0,
            delta_formant_1_hz: 0.0,
            delta_formant_2_hz: 0.0,
            delta_formant_3_hz: 0.0,
            delta_formant_1_bandwidth: 0.0,
            delta_formant_2_bandwidth: 0.0,
            delta_formant_dispersion: 0.0,
            delta_spectral_centroid: 0.0,
            delta_spectral_spread: 0.0,
            delta_spectral_skewness: 0.0,
            delta_spectral_kurtosis: 0.0,
            delta_spectral_tilt: 0.0,
            delta_fm_slope: 0.0,
            delta_am_depth: 0.0,
            delta_subharmonic_ratio: 0.0,
            delta_spectral_entropy: 0.0,
        }
    }

    /// Calculate delta from two vectors (target - source)
    pub fn from_vectors(target: &Vector45D, source: &Vector45D) -> Self {
        let t = target.to_array();
        let s = source.to_array();
        let mut result = [0.0f32; 45];

        for i in 0..45 {
            result[i] = t[i] - s[i];
        }

        Self::from_array(result)
    }

    /// Convert to flat array
    pub fn to_array(&self) -> [f32; 45] {
        let mut arr = [0.0f32; 45];

        // Fundamental (3)
        arr[0] = self.delta_mean_f0_hz;
        arr[1] = self.delta_duration_ms;
        arr[2] = self.delta_f0_range_hz;
        // Grit (3)
        arr[3] = self.delta_harmonic_to_noise_ratio;
        arr[4] = self.delta_spectral_flatness;
        arr[5] = self.delta_harmonicity;
        // Motion (7)
        arr[6] = self.delta_attack_time_ms;
        arr[7] = self.delta_decay_time_ms;
        arr[8] = self.delta_sustain_level;
        arr[9] = self.delta_vibrato_rate_hz;
        arr[10] = self.delta_vibrato_depth;
        arr[11] = self.delta_jitter;
        arr[12] = self.delta_shimmer;
        // Fingerprint (14)
        arr[13] = self.delta_mfcc_1;
        arr[14] = self.delta_mfcc_2;
        arr[15] = self.delta_mfcc_3;
        arr[16] = self.delta_mfcc_4;
        arr[17] = self.delta_mfcc_5;
        arr[18] = self.delta_mfcc_6;
        arr[19] = self.delta_mfcc_7;
        arr[20] = self.delta_mfcc_8;
        arr[21] = self.delta_mfcc_9;
        arr[22] = self.delta_mfcc_10;
        arr[23] = self.delta_mfcc_11;
        arr[24] = self.delta_mfcc_12;
        arr[25] = self.delta_mfcc_13;
        arr[26] = self.delta_spectral_flux;
        // Rhythm (3)
        arr[27] = self.delta_median_ici_ms;
        arr[28] = self.delta_onset_rate_hz;
        arr[29] = self.delta_ici_coefficient_of_variation;
        // Resonance (6)
        arr[30] = self.delta_formant_1_hz;
        arr[31] = self.delta_formant_2_hz;
        arr[32] = self.delta_formant_3_hz;
        arr[33] = self.delta_formant_1_bandwidth;
        arr[34] = self.delta_formant_2_bandwidth;
        arr[35] = self.delta_formant_dispersion;
        // Spectral Shape (4)
        arr[36] = self.delta_spectral_centroid;
        arr[37] = self.delta_spectral_spread;
        arr[38] = self.delta_spectral_skewness;
        arr[39] = self.delta_spectral_kurtosis;
        // Modulation (3)
        arr[40] = self.delta_spectral_tilt;
        arr[41] = self.delta_fm_slope;
        arr[42] = self.delta_am_depth;
        // Non-Linear (2)
        arr[43] = self.delta_subharmonic_ratio;
        arr[44] = self.delta_spectral_entropy;

        arr
    }

    /// Convert from flat array
    pub fn from_array(arr: [f32; 45]) -> Self {
        Self {
            delta_mean_f0_hz: arr[0],
            delta_duration_ms: arr[1],
            delta_f0_range_hz: arr[2],
            delta_harmonic_to_noise_ratio: arr[3],
            delta_spectral_flatness: arr[4],
            delta_harmonicity: arr[5],
            delta_attack_time_ms: arr[6],
            delta_decay_time_ms: arr[7],
            delta_sustain_level: arr[8],
            delta_vibrato_rate_hz: arr[9],
            delta_vibrato_depth: arr[10],
            delta_jitter: arr[11],
            delta_shimmer: arr[12],
            delta_mfcc_1: arr[13],
            delta_mfcc_2: arr[14],
            delta_mfcc_3: arr[15],
            delta_mfcc_4: arr[16],
            delta_mfcc_5: arr[17],
            delta_mfcc_6: arr[18],
            delta_mfcc_7: arr[19],
            delta_mfcc_8: arr[20],
            delta_mfcc_9: arr[21],
            delta_mfcc_10: arr[22],
            delta_mfcc_11: arr[23],
            delta_mfcc_12: arr[24],
            delta_mfcc_13: arr[25],
            delta_spectral_flux: arr[26],
            delta_median_ici_ms: arr[27],
            delta_onset_rate_hz: arr[28],
            delta_ici_coefficient_of_variation: arr[29],
            delta_formant_1_hz: arr[30],
            delta_formant_2_hz: arr[31],
            delta_formant_3_hz: arr[32],
            delta_formant_1_bandwidth: arr[33],
            delta_formant_2_bandwidth: arr[34],
            delta_formant_dispersion: arr[35],
            delta_spectral_centroid: arr[36],
            delta_spectral_spread: arr[37],
            delta_spectral_skewness: arr[38],
            delta_spectral_kurtosis: arr[39],
            delta_spectral_tilt: arr[40],
            delta_fm_slope: arr[41],
            delta_am_depth: arr[42],
            delta_subharmonic_ratio: arr[43],
            delta_spectral_entropy: arr[44],
        }
    }

    /// Scale delta by factor
    pub fn scale(&self, factor: f32) -> Self {
        let d = self.to_array();
        let mut result = [0.0f32; 45];
        for i in 0..45 {
            result[i] = d[i] * factor;
        }
        Self::from_array(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector45d_default() {
        let v = Vector45D::default();
        assert!((v.mean_f0_hz - 7000.0).abs() < 0.01);
        assert!((v.duration_ms - 50.0).abs() < 0.01);
        assert!((v.harmonic_to_noise_ratio - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_vector45d_to_from_array() {
        let v = Vector45D::default();
        let arr = v.to_array();
        assert_eq!(arr.len(), 45);

        let v2 = Vector45D::from_array(arr);
        assert_eq!(v, v2);
    }

    #[test]
    fn test_distance_to_self() {
        let v = Vector45D::default();
        let dist = v.distance_to(&v);
        assert!((dist - 0.0).abs() < 1e-6, "Distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_distance_to_different() {
        let v1 = Vector45D::default();
        let v2 = Vector45D {
            mean_f0_hz: 8000.0,
            ..Default::default()
        };

        let dist = v1.distance_to(&v2);
        assert!(dist > 0.0, "Distance should be positive, got {}", dist);
    }

    #[test]
    fn test_interpolate_alpha_zero() {
        let v1 = Vector45D::default();
        let v2 = Vector45D {
            mean_f0_hz: 8000.0,
            ..Default::default()
        };

        let result = v1.interpolate(&v2, 0.0);
        assert_eq!(result, v1, "Alpha=0 should return first vector");
    }

    #[test]
    fn test_interpolate_alpha_one() {
        let v1 = Vector45D::default();
        let v2 = Vector45D {
            mean_f0_hz: 8000.0,
            ..Default::default()
        };

        let result = v1.interpolate(&v2, 1.0);
        assert_eq!(result, v2, "Alpha=1 should return second vector");
    }

    #[test]
    fn test_interpolate_alpha_half() {
        let v1 = Vector45D {
            mean_f0_hz: 6000.0,
            ..Default::default()
        };
        let v2 = Vector45D {
            mean_f0_hz: 8000.0,
            ..Default::default()
        };

        let result = v1.interpolate(&v2, 0.5);
        assert!(
            (result.mean_f0_hz - 7000.0).abs() < 0.01,
            "Midpoint should be 7000, got {}",
            result.mean_f0_hz
        );
    }

    #[test]
    fn test_extrapolate_unit_factor() {
        let v = Vector45D {
            mean_f0_hz: 7000.0,
            ..Default::default()
        };
        let delta = VectorDelta45D {
            delta_mean_f0_hz: 1000.0,
            ..VectorDelta45D::zero()
        };

        let result = v.extrapolate(&delta, 1.0);
        assert!(
            (result.mean_f0_hz - 8000.0).abs() < 0.01,
            "Extrapolated f0 should be 8000, got {}",
            result.mean_f0_hz
        );
    }

    #[test]
    fn test_extrapolate_double_factor() {
        let v = Vector45D {
            mean_f0_hz: 7000.0,
            ..Default::default()
        };
        let delta = VectorDelta45D {
            delta_mean_f0_hz: 1000.0,
            ..VectorDelta45D::zero()
        };

        let result = v.extrapolate(&delta, 2.0);
        assert!(
            (result.mean_f0_hz - 9000.0).abs() < 0.01,
            "Extrapolated f0 should be 9000, got {}",
            result.mean_f0_hz
        );
    }

    #[test]
    fn test_extrapolate_zero_factor() {
        let v = Vector45D {
            mean_f0_hz: 7000.0,
            ..Default::default()
        };
        let delta = VectorDelta45D {
            delta_mean_f0_hz: 1000.0,
            ..VectorDelta45D::zero()
        };

        let result = v.extrapolate(&delta, 0.0);
        assert_eq!(result, v, "Factor=0 should return original vector");
    }

    #[test]
    fn test_vector_delta_zero() {
        let delta = VectorDelta45D::zero();
        let arr = delta.to_array();
        for (i, &val) in arr.iter().enumerate() {
            assert!((val - 0.0).abs() < 1e-6, "Delta[{}] should be 0, got {}", i, val);
        }
    }

    #[test]
    fn test_vector_delta_from_vectors() {
        let v1 = Vector45D {
            mean_f0_hz: 6000.0,
            ..Default::default()
        };
        let v2 = Vector45D {
            mean_f0_hz: 8000.0,
            ..Default::default()
        };

        let delta = VectorDelta45D::from_vectors(&v2, &v1);
        assert!(
            (delta.delta_mean_f0_hz - 2000.0).abs() < 0.01,
            "Delta should be 2000, got {}",
            delta.delta_mean_f0_hz
        );
    }

    #[test]
    fn test_vector_add() {
        let v1 = Vector45D {
            mean_f0_hz: 6000.0,
            ..Default::default()
        };
        let v2 = Vector45D {
            mean_f0_hz: 2000.0,
            ..Default::default()
        };

        let result = v1 + v2;
        assert!(
            (result.mean_f0_hz - 8000.0).abs() < 0.01,
            "Sum should be 8000, got {}",
            result.mean_f0_hz
        );
    }

    #[test]
    fn test_vector_sub() {
        let v1 = Vector45D {
            mean_f0_hz: 8000.0,
            ..Default::default()
        };
        let v2 = Vector45D {
            mean_f0_hz: 2000.0,
            ..Default::default()
        };

        let result = v1 - v2;
        assert!(
            (result.mean_f0_hz - 6000.0).abs() < 0.01,
            "Difference should be 6000, got {}",
            result.mean_f0_hz
        );
    }

    #[test]
    fn test_vector_scale() {
        let v = Vector45D {
            mean_f0_hz: 7000.0,
            ..Default::default()
        };

        let result = v * 2.0;
        assert!(
            (result.mean_f0_hz - 14000.0).abs() < 0.01,
            "Scaled f0 should be 14000, got {}",
            result.mean_f0_hz
        );
    }

    #[test]
    fn test_magnitude() {
        let v = Vector45D::default();
        let mag = v.magnitude();
        assert!(mag >= 0.0, "Magnitude should be non-negative, got {}", mag);
    }

    #[test]
    fn test_normalized() {
        let v = Vector45D {
            mean_f0_hz: 7000.0,
            ..Default::default()
        };

        let normalized = v.normalized();
        let mag = normalized.magnitude();
        assert!(
            (mag - 1.0).abs() < 0.01 || mag < 1e-6,
            "Normalized magnitude should be 1, got {}",
            mag
        );
    }

    #[test]
    fn test_feature_weights_sum() {
        let weights = Vector45D::feature_weights();
        assert_eq!(weights.len(), 45, "Should have 45 weights");

        // Check critical weights are higher
        assert!(weights[0] > weights[11], "F0 weight should be higher than jitter");
        assert!(
            weights[9] > weights[11],
            "Vibrato rate weight should be higher than jitter"
        );
    }

    #[test]
    fn test_normalization_ranges_positive() {
        let ranges = Vector45D::normalization_ranges();
        for (i, &range) in ranges.iter().enumerate() {
            assert!(range > 0.0, "Range[{}] should be positive, got {}", i, range);
        }
    }

    #[test]
    fn test_all_45_dimensions_in_array() {
        let v = Vector45D::default();
        let arr = v.to_array();

        // Verify all 45 values are accessible
        assert_eq!(arr.len(), 45);

        // Verify specific indices
        assert!((arr[0] - v.mean_f0_hz).abs() < 0.01); // Fundamental
        assert!((arr[3] - v.harmonic_to_noise_ratio).abs() < 0.01); // Grit
        assert!((arr[6] - v.attack_time_ms).abs() < 0.01); // Motion
        assert!((arr[13] - v.mfcc_1).abs() < 0.01); // Fingerprint
        assert!((arr[27] - v.median_ici_ms).abs() < 0.01); // Rhythm
        assert!((arr[30] - v.formant_1_hz).abs() < 0.01); // Resonance
        assert!((arr[36] - v.spectral_centroid).abs() < 0.01); // Spectral Shape
        assert!((arr[40] - v.spectral_tilt).abs() < 0.01); // Modulation
        assert!((arr[43] - v.subharmonic_ratio).abs() < 0.01); // Non-Linear
    }
}

// ============================================================================
// PYTHON BINDINGS (PyO3)
// ============================================================================

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Python wrapper for Vector45D
#[cfg(feature = "python-bindings")]
#[pyclass(name = "Vector45D")]
#[derive(Clone)]
pub struct PyVector45D {
    pub inner: Vector45D,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVector45D {
    #[new]
    fn new() -> Self {
        Self {
            inner: Vector45D::default(),
        }
    }

    /// Create from flat array
    #[staticmethod]
    fn from_array_py(arr: Vec<f32>) -> PyResult<Self> {
        if arr.len() != 45 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Array must have 45 elements, got {}",
                arr.len()
            )));
        }
        let mut fixed_arr = [0.0f32; 45];
        fixed_arr.copy_from_slice(&arr);
        Ok(Self {
            inner: Vector45D::from_array(fixed_arr),
        })
    }

    /// Convert to flat array
    fn to_array_py(&self) -> Vec<f32> {
        self.inner.to_array().to_vec()
    }

    /// Calculate distance to another vector
    fn distance_to(&self, other: &PyVector45D) -> f32 {
        self.inner.distance_to(&other.inner)
    }

    /// Interpolate to another vector
    fn interpolate(&self, other: &PyVector45D, alpha: f32) -> PyVector45D {
        PyVector45D {
            inner: self.inner.interpolate(&other.inner, alpha),
        }
    }

    /// Get mean F0
    #[getter]
    fn mean_f0_hz(&self) -> f32 {
        self.inner.mean_f0_hz
    }

    /// Set mean F0
    #[setter]
    fn set_mean_f0_hz(&mut self, value: f32) {
        self.inner.mean_f0_hz = value;
    }

    /// Get duration
    #[getter]
    fn duration_ms(&self) -> f32 {
        self.inner.duration_ms
    }

    /// Set duration
    #[setter]
    fn set_duration_ms(&mut self, value: f32) {
        self.inner.duration_ms = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "Vector45D(f0={:.1}Hz, duration={:.1}ms)",
            self.inner.mean_f0_hz, self.inner.duration_ms
        )
    }
}

/// Python wrapper for VectorDelta45D
#[cfg(feature = "python-bindings")]
#[pyclass(name = "VectorDelta45D")]
#[derive(Clone)]
pub struct PyVectorDelta45D {
    pub inner: VectorDelta45D,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyVectorDelta45D {
    #[new]
    fn new() -> Self {
        Self {
            inner: VectorDelta45D::zero(),
        }
    }

    /// Create from two vectors
    #[staticmethod]
    fn from_vectors(target: &PyVector45D, source: &PyVector45D) -> Self {
        Self {
            inner: VectorDelta45D::from_vectors(&target.inner, &source.inner),
        }
    }

    /// Convert to flat array
    fn to_array_py(&self) -> Vec<f32> {
        self.inner.to_array().to_vec()
    }
}
