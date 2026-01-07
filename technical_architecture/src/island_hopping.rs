//! Island Hopping Navigation Module (Rust Execution Layer)
//! =======================================================
//!
//! This module implements the execution layer for Island Hopping navigation strategy.
//! It provides high-performance, safety-critical components for:
//!
//! - 30D Vector Math Operations (SIMD-optimized)
//! - Delta Clamping (Safety-critical)
//! - Nearest Neighbor Lookup (KD-tree indexed)
//! - Timeline Orchestration (Real-time)
//! - Granular Delta Application
//!
//! Architecture: Execution vs. Logic Split
//! ----------------------------------------
//! - **Execution Layer (Rust - this file)**: Time-critical operations, safety
//! - **Logic Layer (Python)**: Cognitive intelligence, decision making, learning
//!
//! Vector Space Dimensions:
//! ------------------------
//! The 30D feature vector includes:
//! - Fundamental (3): mean_f0_hz, f0_range_hz, duration_ms
//! - Grit Factors (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
//! - Motion Factors (7): attack_time_ms, decay_time_ms, sustain_level,
//!   vibrato_rate_hz, vibrato_depth, jitter, shimmer
//! - Fingerprint Factors (13 MFCCs): mfcc_1 through mfcc_13
//! - Spectral Dynamics (1): spectral_flux
//! - Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use log::warn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// 30D Vector Math Operations (Priority 1: Critical)
// ============================================================================

/// 30-dimensional acoustic feature vector
///
/// Features organized by category:
/// - Fundamental (3): mean_f0_hz, f0_range_hz, duration_ms
/// - Grit Factors (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
/// - Motion Factors (7): attack_time_ms, decay_time_ms, sustain_level, vibrato_rate_hz, vibrato_depth, jitter, shimmer
/// - Fingerprint Factors (13 MFCCs): mfcc_1 through mfcc_13
/// - Spectral Dynamics (1): spectral_flux
/// - Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector30D {
    // === Fundamental (3 features) ===
    pub mean_f0_hz: f32,
    pub f0_range_hz: f32,
    pub duration_ms: f32,

    // === Grit Factors (3 features) ===
    pub harmonic_to_noise_ratio: f32,
    pub spectral_flatness: f32,
    pub harmonicity: f32,

    // === Motion Factors (7 features) ===
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    pub sustain_level: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth: f32,
    pub jitter: f32,
    pub shimmer: f32,

    // === Fingerprint Factors (13 MFCCs) ===
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

    // === Spectral Dynamics (1 feature) ===
    pub spectral_flux: f32,

    // === Rhythm Factors (3 features) ===
    pub median_ici_ms: f32,
    pub onset_rate_hz: f32,
    pub ici_coefficient_of_variation: f32,
}

impl Default for Vector30D {
    fn default() -> Self {
        Self {
            // Fundamental (3)
            mean_f0_hz: 7000.0,
            f0_range_hz: 400.0,
            duration_ms: 50.0,
            // Grit Factors (3)
            harmonic_to_noise_ratio: 20.0,
            spectral_flatness: 0.3,
            harmonicity: 0.8,
            // Motion Factors (7)
            attack_time_ms: 5.0,
            decay_time_ms: 20.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 7.0,
            vibrato_depth: 0.02,
            jitter: 0.01,
            shimmer: 0.03,
            // Fingerprint Factors (13 MFCCs)
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
            // Spectral Dynamics (1)
            spectral_flux: 0.5,
            // Rhythm Factors (3)
            median_ici_ms: 15.0,
            onset_rate_hz: 8.0,
            ici_coefficient_of_variation: 0.3,
        }
    }
}

impl Vector30D {
    /// Create a new Vector30D with all 30 dimensions
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // Fundamental (3)
        mean_f0_hz: f32,
        f0_range_hz: f32,
        duration_ms: f32,
        // Grit Factors (3)
        harmonic_to_noise_ratio: f32,
        spectral_flatness: f32,
        harmonicity: f32,
        // Motion Factors (7)
        attack_time_ms: f32,
        decay_time_ms: f32,
        sustain_level: f32,
        vibrato_rate_hz: f32,
        vibrato_depth: f32,
        jitter: f32,
        shimmer: f32,
        // Fingerprint Factors (13 MFCCs)
        mfcc_1: f32,
        mfcc_2: f32,
        mfcc_3: f32,
        mfcc_4: f32,
        mfcc_5: f32,
        mfcc_6: f32,
        mfcc_7: f32,
        mfcc_8: f32,
        mfcc_9: f32,
        mfcc_10: f32,
        mfcc_11: f32,
        mfcc_12: f32,
        mfcc_13: f32,
        // Spectral Dynamics (1)
        spectral_flux: f32,
        // Rhythm Factors (3)
        median_ici_ms: f32,
        onset_rate_hz: f32,
        ici_coefficient_of_variation: f32,
    ) -> Self {
        Self {
            mean_f0_hz,
            f0_range_hz,
            duration_ms,
            harmonic_to_noise_ratio,
            spectral_flatness,
            harmonicity,
            attack_time_ms,
            decay_time_ms,
            sustain_level,
            vibrato_rate_hz,
            vibrato_depth,
            jitter,
            shimmer,
            mfcc_1,
            mfcc_2,
            mfcc_3,
            mfcc_4,
            mfcc_5,
            mfcc_6,
            mfcc_7,
            mfcc_8,
            mfcc_9,
            mfcc_10,
            mfcc_11,
            mfcc_12,
            mfcc_13,
            spectral_flux,
            median_ici_ms,
            onset_rate_hz,
            ici_coefficient_of_variation,
        }
    }

    /// Convert to array for SIMD operations
    pub fn to_array(&self) -> [f32; 30] {
        [
            // Fundamental (3)
            self.mean_f0_hz,
            self.f0_range_hz,
            self.duration_ms,
            // Grit Factors (3)
            self.harmonic_to_noise_ratio,
            self.spectral_flatness,
            self.harmonicity,
            // Motion Factors (7)
            self.attack_time_ms,
            self.decay_time_ms,
            self.sustain_level,
            self.vibrato_rate_hz,
            self.vibrato_depth,
            self.jitter,
            self.shimmer,
            // Fingerprint Factors (13 MFCCs)
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
            // Spectral Dynamics (1)
            self.spectral_flux,
            // Rhythm Factors (3)
            self.median_ici_ms,
            self.onset_rate_hz,
            self.ici_coefficient_of_variation,
        ]
    }

    /// Convert from array
    pub fn from_array(arr: [f32; 30]) -> Self {
        Self {
            mean_f0_hz: arr[0],
            f0_range_hz: arr[1],
            duration_ms: arr[2],
            harmonic_to_noise_ratio: arr[3],
            spectral_flatness: arr[4],
            harmonicity: arr[5],
            attack_time_ms: arr[6],
            decay_time_ms: arr[7],
            sustain_level: arr[8],
            vibrato_rate_hz: arr[9],
            vibrato_depth: arr[10],
            jitter: arr[11],
            shimmer: arr[12],
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
            median_ici_ms: arr[27],
            onset_rate_hz: arr[28],
            ici_coefficient_of_variation: arr[29],
        }
    }

    /// Get normalization ranges for each dimension
    fn normalization_ranges() -> [f32; 30] {
        [
            // Fundamental (3)
            2000.0, // mean_f0_hz: 0-2000 Hz typical range
            500.0,  // f0_range_hz: 0-500 Hz
            50.0,   // duration_ms: 0-50 ms typical range
            // Grit Factors (3)
            30.0, // harmonic_to_noise_ratio: 0-30 dB
            1.0,  // spectral_flatness: 0-1
            1.0,  // harmonicity: 0-1
            // Motion Factors (7)
            20.0, // attack_time_ms: 0-20 ms
            50.0, // decay_time_ms: 0-50 ms
            1.0,  // sustain_level: 0-1
            20.0, // vibrato_rate_hz: 0-20 Hz
            0.1,  // vibrato_depth: 0-0.1
            0.05, // jitter: 0-0.05
            0.1,  // shimmer: 0-0.1
            // Fingerprint Factors (13 MFCCs)
            20.0, // mfcc_1: -20 to 0
            20.0, // mfcc_2: -20 to 0
            20.0, // mfcc_3: -20 to 0
            20.0, // mfcc_4: -20 to 0
            20.0, // mfcc_5: -20 to 0
            20.0, // mfcc_6: -20 to 0
            20.0, // mfcc_7: -20 to 0
            20.0, // mfcc_8: -20 to 0
            20.0, // mfcc_9: -20 to 0
            20.0, // mfcc_10: -20 to 0
            20.0, // mfcc_11: -20 to 0
            20.0, // mfcc_12: -20 to 0
            20.0, // mfcc_13: -20 to 0
            // Spectral Dynamics (1)
            1.0, // spectral_flux: 0-1
            // Rhythm Factors (3)
            200.0, // median_ici_ms: 0-200 ms
            20.0,  // onset_rate_hz: 0-20 Hz
            1.0,   // ici_coefficient_of_variation: 0-1
        ]
    }

    /// Calculate normalized Euclidean distance to another vector
    ///
    /// This is the PRIMARY distance metric for Island Hopping navigation.
    /// Distances are normalized by dimension-specific ranges to ensure
    /// meaningful comparisons across different acoustic features.
    ///
    /// Uses the 30D core dimensions for distance calculation.
    pub fn distance_to(&self, other: &Vector30D) -> f32 {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let ranges = Self::normalization_ranges();

        let mut sum_squared = 0.0_f32;
        for i in 0..30 {
            let diff = (v1[i] - v2[i]) / ranges[i];
            sum_squared += diff * diff;
        }

        sum_squared.sqrt()
    }

    /// Linear interpolation between two vectors (Bridge Builder)
    ///
    /// This is SAFE navigation between two known islands.
    /// Alpha must be in [0.0, 1.0]:
    /// - 0.0 = return self
    /// - 0.5 = midpoint
    /// - 1.0 = return other
    pub fn interpolate(&self, other: &Vector30D, alpha: f32) -> Vector30D {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be in [0, 1], got {}",
            alpha
        );

        Vector30D {
            mean_f0_hz: self.mean_f0_hz * (1.0 - alpha) + other.mean_f0_hz * alpha,
            f0_range_hz: self.f0_range_hz * (1.0 - alpha) + other.f0_range_hz * alpha,
            duration_ms: self.duration_ms * (1.0 - alpha) + other.duration_ms * alpha,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio * (1.0 - alpha)
                + other.harmonic_to_noise_ratio * alpha,
            spectral_flatness: self.spectral_flatness * (1.0 - alpha)
                + other.spectral_flatness * alpha,
            harmonicity: self.harmonicity * (1.0 - alpha) + other.harmonicity * alpha,
            attack_time_ms: self.attack_time_ms * (1.0 - alpha) + other.attack_time_ms * alpha,
            decay_time_ms: self.decay_time_ms * (1.0 - alpha) + other.decay_time_ms * alpha,
            sustain_level: self.sustain_level * (1.0 - alpha) + other.sustain_level * alpha,
            vibrato_rate_hz: self.vibrato_rate_hz * (1.0 - alpha) + other.vibrato_rate_hz * alpha,
            vibrato_depth: self.vibrato_depth * (1.0 - alpha) + other.vibrato_depth * alpha,
            jitter: self.jitter * (1.0 - alpha) + other.jitter * alpha,
            shimmer: self.shimmer * (1.0 - alpha) + other.shimmer * alpha,
            mfcc_1: self.mfcc_1 * (1.0 - alpha) + other.mfcc_1 * alpha,
            mfcc_2: self.mfcc_2 * (1.0 - alpha) + other.mfcc_2 * alpha,
            mfcc_3: self.mfcc_3 * (1.0 - alpha) + other.mfcc_3 * alpha,
            mfcc_4: self.mfcc_4 * (1.0 - alpha) + other.mfcc_4 * alpha,
            mfcc_5: self.mfcc_5 * (1.0 - alpha) + other.mfcc_5 * alpha,
            mfcc_6: self.mfcc_6 * (1.0 - alpha) + other.mfcc_6 * alpha,
            mfcc_7: self.mfcc_7 * (1.0 - alpha) + other.mfcc_7 * alpha,
            mfcc_8: self.mfcc_8 * (1.0 - alpha) + other.mfcc_8 * alpha,
            mfcc_9: self.mfcc_9 * (1.0 - alpha) + other.mfcc_9 * alpha,
            mfcc_10: self.mfcc_10 * (1.0 - alpha) + other.mfcc_10 * alpha,
            mfcc_11: self.mfcc_11 * (1.0 - alpha) + other.mfcc_11 * alpha,
            mfcc_12: self.mfcc_12 * (1.0 - alpha) + other.mfcc_12 * alpha,
            mfcc_13: self.mfcc_13 * (1.0 - alpha) + other.mfcc_13 * alpha,
            spectral_flux: self.spectral_flux * (1.0 - alpha) + other.spectral_flux * alpha,
            median_ici_ms: self.median_ici_ms * (1.0 - alpha) + other.median_ici_ms * alpha,
            onset_rate_hz: self.onset_rate_hz * (1.0 - alpha) + other.onset_rate_hz * alpha,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation * (1.0 - alpha)
                + other.ici_coefficient_of_variation * alpha,
        }
    }

    /// Vector extrapolation beyond origin (Ocean Explorer)
    ///
    /// This is RISKY navigation beyond known islands into "open ocean".
    /// Factor must be >= 0.0:
    /// - 0.0 = return origin (no movement)
    /// - 1.0 = move to origin + direction
    /// - 2.0 = move twice as far in direction
    pub fn extrapolate(&self, direction: &VectorDelta, factor: f32) -> Vector30D {
        assert!(factor >= 0.0, "Factor must be >= 0, got {}", factor);

        Vector30D {
            mean_f0_hz: self.mean_f0_hz + direction.delta_mean_f0_hz * factor,
            f0_range_hz: self.f0_range_hz + direction.delta_f0_range_hz * factor,
            duration_ms: self.duration_ms + direction.delta_duration_ms * factor,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio
                + direction.delta_harmonic_to_noise_ratio * factor,
            spectral_flatness: self.spectral_flatness + direction.delta_spectral_flatness * factor,
            harmonicity: self.harmonicity + direction.delta_harmonicity * factor,
            attack_time_ms: self.attack_time_ms + direction.delta_attack_time_ms * factor,
            decay_time_ms: self.decay_time_ms + direction.delta_decay_time_ms * factor,
            sustain_level: self.sustain_level + direction.delta_sustain_level * factor,
            vibrato_rate_hz: self.vibrato_rate_hz + direction.delta_vibrato_rate_hz * factor,
            vibrato_depth: self.vibrato_depth + direction.delta_vibrato_depth * factor,
            jitter: self.jitter + direction.delta_jitter * factor,
            shimmer: self.shimmer + direction.delta_shimmer * factor,
            mfcc_1: self.mfcc_1 + direction.delta_mfcc_1 * factor,
            mfcc_2: self.mfcc_2 + direction.delta_mfcc_2 * factor,
            mfcc_3: self.mfcc_3 + direction.delta_mfcc_3 * factor,
            mfcc_4: self.mfcc_4 + direction.delta_mfcc_4 * factor,
            mfcc_5: self.mfcc_5 + direction.delta_mfcc_5 * factor,
            mfcc_6: self.mfcc_6 + direction.delta_mfcc_6 * factor,
            mfcc_7: self.mfcc_7 + direction.delta_mfcc_7 * factor,
            mfcc_8: self.mfcc_8 + direction.delta_mfcc_8 * factor,
            mfcc_9: self.mfcc_9 + direction.delta_mfcc_9 * factor,
            mfcc_10: self.mfcc_10 + direction.delta_mfcc_10 * factor,
            mfcc_11: self.mfcc_11 + direction.delta_mfcc_11 * factor,
            mfcc_12: self.mfcc_12 + direction.delta_mfcc_12 * factor,
            mfcc_13: self.mfcc_13 + direction.delta_mfcc_13 * factor,
            spectral_flux: self.spectral_flux + direction.delta_spectral_flux * factor,
            median_ici_ms: self.median_ici_ms + direction.delta_median_ici_ms * factor,
            onset_rate_hz: self.onset_rate_hz + direction.delta_onset_rate_hz * factor,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation
                + direction.delta_ici_coefficient_of_variation * factor,
        }
    }

    /// Add two vectors (for delta operations)
    pub fn add(&self, other: &Vector30D) -> Vector30D {
        Vector30D {
            mean_f0_hz: self.mean_f0_hz + other.mean_f0_hz,
            f0_range_hz: self.f0_range_hz + other.f0_range_hz,
            duration_ms: self.duration_ms + other.duration_ms,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio + other.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness + other.spectral_flatness,
            harmonicity: self.harmonicity + other.harmonicity,
            attack_time_ms: self.attack_time_ms + other.attack_time_ms,
            decay_time_ms: self.decay_time_ms + other.decay_time_ms,
            sustain_level: self.sustain_level + other.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz + other.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth + other.vibrato_depth,
            jitter: self.jitter + other.jitter,
            shimmer: self.shimmer + other.shimmer,
            mfcc_1: self.mfcc_1 + other.mfcc_1,
            mfcc_2: self.mfcc_2 + other.mfcc_2,
            mfcc_3: self.mfcc_3 + other.mfcc_3,
            mfcc_4: self.mfcc_4 + other.mfcc_4,
            mfcc_5: self.mfcc_5 + other.mfcc_5,
            mfcc_6: self.mfcc_6 + other.mfcc_6,
            mfcc_7: self.mfcc_7 + other.mfcc_7,
            mfcc_8: self.mfcc_8 + other.mfcc_8,
            mfcc_9: self.mfcc_9 + other.mfcc_9,
            mfcc_10: self.mfcc_10 + other.mfcc_10,
            mfcc_11: self.mfcc_11 + other.mfcc_11,
            mfcc_12: self.mfcc_12 + other.mfcc_12,
            mfcc_13: self.mfcc_13 + other.mfcc_13,
            spectral_flux: self.spectral_flux + other.spectral_flux,
            median_ici_ms: self.median_ici_ms + other.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz + other.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation
                + other.ici_coefficient_of_variation,
        }
    }

    /// Subtract two vectors (for delta calculation)
    pub fn sub(&self, other: &Vector30D) -> Vector30D {
        Vector30D {
            mean_f0_hz: self.mean_f0_hz - other.mean_f0_hz,
            f0_range_hz: self.f0_range_hz - other.f0_range_hz,
            duration_ms: self.duration_ms - other.duration_ms,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio - other.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness - other.spectral_flatness,
            harmonicity: self.harmonicity - other.harmonicity,
            attack_time_ms: self.attack_time_ms - other.attack_time_ms,
            decay_time_ms: self.decay_time_ms - other.decay_time_ms,
            sustain_level: self.sustain_level - other.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz - other.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth - other.vibrato_depth,
            jitter: self.jitter - other.jitter,
            shimmer: self.shimmer - other.shimmer,
            mfcc_1: self.mfcc_1 - other.mfcc_1,
            mfcc_2: self.mfcc_2 - other.mfcc_2,
            mfcc_3: self.mfcc_3 - other.mfcc_3,
            mfcc_4: self.mfcc_4 - other.mfcc_4,
            mfcc_5: self.mfcc_5 - other.mfcc_5,
            mfcc_6: self.mfcc_6 - other.mfcc_6,
            mfcc_7: self.mfcc_7 - other.mfcc_7,
            mfcc_8: self.mfcc_8 - other.mfcc_8,
            mfcc_9: self.mfcc_9 - other.mfcc_9,
            mfcc_10: self.mfcc_10 - other.mfcc_10,
            mfcc_11: self.mfcc_11 - other.mfcc_11,
            mfcc_12: self.mfcc_12 - other.mfcc_12,
            mfcc_13: self.mfcc_13 - other.mfcc_13,
            spectral_flux: self.spectral_flux - other.spectral_flux,
            median_ici_ms: self.median_ici_ms - other.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz - other.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation
                - other.ici_coefficient_of_variation,
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, factor: f32) -> Vector30D {
        Vector30D {
            mean_f0_hz: self.mean_f0_hz * factor,
            f0_range_hz: self.f0_range_hz * factor,
            duration_ms: self.duration_ms * factor,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio * factor,
            spectral_flatness: self.spectral_flatness * factor,
            harmonicity: self.harmonicity * factor,
            attack_time_ms: self.attack_time_ms * factor,
            decay_time_ms: self.decay_time_ms * factor,
            sustain_level: self.sustain_level * factor,
            vibrato_rate_hz: self.vibrato_rate_hz * factor,
            vibrato_depth: self.vibrato_depth * factor,
            jitter: self.jitter * factor,
            shimmer: self.shimmer * factor,
            mfcc_1: self.mfcc_1 * factor,
            mfcc_2: self.mfcc_2 * factor,
            mfcc_3: self.mfcc_3 * factor,
            mfcc_4: self.mfcc_4 * factor,
            mfcc_5: self.mfcc_5 * factor,
            mfcc_6: self.mfcc_6 * factor,
            mfcc_7: self.mfcc_7 * factor,
            mfcc_8: self.mfcc_8 * factor,
            mfcc_9: self.mfcc_9 * factor,
            mfcc_10: self.mfcc_10 * factor,
            mfcc_11: self.mfcc_11 * factor,
            mfcc_12: self.mfcc_12 * factor,
            mfcc_13: self.mfcc_13 * factor,
            spectral_flux: self.spectral_flux * factor,
            median_ici_ms: self.median_ici_ms * factor,
            onset_rate_hz: self.onset_rate_hz * factor,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation * factor,
        }
    }

    /// Calculate magnitude (normalized Euclidean norm)
    pub fn magnitude(&self) -> f32 {
        let arr = self.to_array();
        let ranges = Self::normalization_ranges();

        let mut sum_squared = 0.0_f32;
        for i in 0..30 {
            let normalized = arr[i] / ranges[i];
            sum_squared += normalized * normalized;
        }

        sum_squared.sqrt()
    }

    /// Normalize to unit vector
    pub fn normalized(&self) -> Vector30D {
        let mag = self.magnitude();
        if mag > 1e-6 {
            self.scale(1.0 / mag)
        } else {
            *self
        }
    }
}

impl std::ops::Add<Vector30D> for Vector30D {
    type Output = Vector30D;

    fn add(self, rhs: Vector30D) -> Self::Output {
        Vector30D {
            mean_f0_hz: self.mean_f0_hz + rhs.mean_f0_hz,
            f0_range_hz: self.f0_range_hz + rhs.f0_range_hz,
            duration_ms: self.duration_ms + rhs.duration_ms,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio + rhs.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness + rhs.spectral_flatness,
            harmonicity: self.harmonicity + rhs.harmonicity,
            attack_time_ms: self.attack_time_ms + rhs.attack_time_ms,
            decay_time_ms: self.decay_time_ms + rhs.decay_time_ms,
            sustain_level: self.sustain_level + rhs.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz + rhs.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth + rhs.vibrato_depth,
            jitter: self.jitter + rhs.jitter,
            shimmer: self.shimmer + rhs.shimmer,
            mfcc_1: self.mfcc_1 + rhs.mfcc_1,
            mfcc_2: self.mfcc_2 + rhs.mfcc_2,
            mfcc_3: self.mfcc_3 + rhs.mfcc_3,
            mfcc_4: self.mfcc_4 + rhs.mfcc_4,
            mfcc_5: self.mfcc_5 + rhs.mfcc_5,
            mfcc_6: self.mfcc_6 + rhs.mfcc_6,
            mfcc_7: self.mfcc_7 + rhs.mfcc_7,
            mfcc_8: self.mfcc_8 + rhs.mfcc_8,
            mfcc_9: self.mfcc_9 + rhs.mfcc_9,
            mfcc_10: self.mfcc_10 + rhs.mfcc_10,
            mfcc_11: self.mfcc_11 + rhs.mfcc_11,
            mfcc_12: self.mfcc_12 + rhs.mfcc_12,
            mfcc_13: self.mfcc_13 + rhs.mfcc_13,
            spectral_flux: self.spectral_flux + rhs.spectral_flux,
            median_ici_ms: self.median_ici_ms + rhs.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz + rhs.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation
                + rhs.ici_coefficient_of_variation,
        }
    }
}

impl std::ops::Sub<Vector30D> for Vector30D {
    type Output = Vector30D;

    fn sub(self, rhs: Vector30D) -> Self::Output {
        Vector30D {
            mean_f0_hz: self.mean_f0_hz - rhs.mean_f0_hz,
            f0_range_hz: self.f0_range_hz - rhs.f0_range_hz,
            duration_ms: self.duration_ms - rhs.duration_ms,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio - rhs.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness - rhs.spectral_flatness,
            harmonicity: self.harmonicity - rhs.harmonicity,
            attack_time_ms: self.attack_time_ms - rhs.attack_time_ms,
            decay_time_ms: self.decay_time_ms - rhs.decay_time_ms,
            sustain_level: self.sustain_level - rhs.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz - rhs.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth - rhs.vibrato_depth,
            jitter: self.jitter - rhs.jitter,
            shimmer: self.shimmer - rhs.shimmer,
            mfcc_1: self.mfcc_1 - rhs.mfcc_1,
            mfcc_2: self.mfcc_2 - rhs.mfcc_2,
            mfcc_3: self.mfcc_3 - rhs.mfcc_3,
            mfcc_4: self.mfcc_4 - rhs.mfcc_4,
            mfcc_5: self.mfcc_5 - rhs.mfcc_5,
            mfcc_6: self.mfcc_6 - rhs.mfcc_6,
            mfcc_7: self.mfcc_7 - rhs.mfcc_7,
            mfcc_8: self.mfcc_8 - rhs.mfcc_8,
            mfcc_9: self.mfcc_9 - rhs.mfcc_9,
            mfcc_10: self.mfcc_10 - rhs.mfcc_10,
            mfcc_11: self.mfcc_11 - rhs.mfcc_11,
            mfcc_12: self.mfcc_12 - rhs.mfcc_12,
            mfcc_13: self.mfcc_13 - rhs.mfcc_13,
            spectral_flux: self.spectral_flux - rhs.spectral_flux,
            median_ici_ms: self.median_ici_ms - rhs.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz - rhs.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation
                - rhs.ici_coefficient_of_variation,
        }
    }
}

impl std::ops::Mul<f32> for Vector30D {
    type Output = Vector30D;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

// ============================================================================
// Vector Delta (30D Difference Vector)
// ============================================================================

/// 30-dimensional delta vector for extrapolation operations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VectorDelta {
    // === Fundamental (3) ===
    pub delta_mean_f0_hz: f32,
    pub delta_f0_range_hz: f32,
    pub delta_duration_ms: f32,

    // === Grit Factors (3) ===
    pub delta_harmonic_to_noise_ratio: f32,
    pub delta_spectral_flatness: f32,
    pub delta_harmonicity: f32,

    // === Motion Factors (7) ===
    pub delta_attack_time_ms: f32,
    pub delta_decay_time_ms: f32,
    pub delta_sustain_level: f32,
    pub delta_vibrato_rate_hz: f32,
    pub delta_vibrato_depth: f32,
    pub delta_jitter: f32,
    pub delta_shimmer: f32,

    // === Fingerprint Factors (13 MFCCs) ===
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

    // === Spectral Dynamics (1) ===
    pub delta_spectral_flux: f32,

    // === Rhythm Factors (3) ===
    pub delta_median_ici_ms: f32,
    pub delta_onset_rate_hz: f32,
    pub delta_ici_coefficient_of_variation: f32,
}

impl VectorDelta {
    /// Create a zero delta (no change)
    pub fn zero() -> Self {
        Self {
            delta_mean_f0_hz: 0.0,
            delta_f0_range_hz: 0.0,
            delta_duration_ms: 0.0,
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
        }
    }

    /// Calculate delta from two vectors (target - source)
    pub fn from_vectors(target: &Vector30D, source: &Vector30D) -> Self {
        Self {
            delta_mean_f0_hz: target.mean_f0_hz - source.mean_f0_hz,
            delta_f0_range_hz: target.f0_range_hz - source.f0_range_hz,
            delta_duration_ms: target.duration_ms - source.duration_ms,
            delta_harmonic_to_noise_ratio: target.harmonic_to_noise_ratio
                - source.harmonic_to_noise_ratio,
            delta_spectral_flatness: target.spectral_flatness - source.spectral_flatness,
            delta_harmonicity: target.harmonicity - source.harmonicity,
            delta_attack_time_ms: target.attack_time_ms - source.attack_time_ms,
            delta_decay_time_ms: target.decay_time_ms - source.decay_time_ms,
            delta_sustain_level: target.sustain_level - source.sustain_level,
            delta_vibrato_rate_hz: target.vibrato_rate_hz - source.vibrato_rate_hz,
            delta_vibrato_depth: target.vibrato_depth - source.vibrato_depth,
            delta_jitter: target.jitter - source.jitter,
            delta_shimmer: target.shimmer - source.shimmer,
            delta_mfcc_1: target.mfcc_1 - source.mfcc_1,
            delta_mfcc_2: target.mfcc_2 - source.mfcc_2,
            delta_mfcc_3: target.mfcc_3 - source.mfcc_3,
            delta_mfcc_4: target.mfcc_4 - source.mfcc_4,
            delta_mfcc_5: target.mfcc_5 - source.mfcc_5,
            delta_mfcc_6: target.mfcc_6 - source.mfcc_6,
            delta_mfcc_7: target.mfcc_7 - source.mfcc_7,
            delta_mfcc_8: target.mfcc_8 - source.mfcc_8,
            delta_mfcc_9: target.mfcc_9 - source.mfcc_9,
            delta_mfcc_10: target.mfcc_10 - source.mfcc_10,
            delta_mfcc_11: target.mfcc_11 - source.mfcc_11,
            delta_mfcc_12: target.mfcc_12 - source.mfcc_12,
            delta_mfcc_13: target.mfcc_13 - source.mfcc_13,
            delta_spectral_flux: target.spectral_flux - source.spectral_flux,
            delta_median_ici_ms: target.median_ici_ms - source.median_ici_ms,
            delta_onset_rate_hz: target.onset_rate_hz - source.onset_rate_hz,
            delta_ici_coefficient_of_variation: target.ici_coefficient_of_variation
                - source.ici_coefficient_of_variation,
        }
    }
}

// ============================================================================
// Delta Clamping (Priority 2: Safety-Critical)
// ============================================================================

/// Navigation waypoint result with clamping information
#[derive(Debug, Clone, PartialEq)]
pub struct NavigationWaypoint {
    /// The (possibly clamped) target vector
    pub target: Vector30D,
    /// Navigation mode used
    pub mode: NavigationMode,
    /// Anchor island if interpolation was used
    pub anchor_island: Option<String>,
    /// Distance from anchor to target (normalized)
    pub distance_to_anchor: f32,
    /// Whether clamping was applied
    pub was_clamped: bool,
    /// Original target before clamping (if clamped)
    pub original_target: Option<Vector30D>,
}

/// Navigation mode classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationMode {
    /// Safe interpolation between known islands
    Interpolation,
    /// Risky extrapolation beyond known islands
    Extrapolation,
    /// Extrapolation that was clamped to safe distance
    ExtrapolationClamped,
}

/// Safety clamp for preventing over-warping artifacts ("The Leash")
///
/// This is a SAFETY-CRITICAL component that prevents synthesis artifacts
/// by limiting the maximum warp distance in 30D space.
#[derive(Debug, Clone)]
pub struct SafetyClamp {
    /// Maximum safe warp distance (normalized)
    max_safe_warp: f32,
}

impl SafetyClamp {
    /// Create a new safety clamp with default 20% max warp
    pub fn new() -> Self {
        Self { max_safe_warp: 0.2 }
    }

    /// Create with custom max warp distance
    pub fn with_max_warp(max_safe_warp: f32) -> Self {
        Self {
            max_safe_warp: max_safe_warp.clamp(0.0, 1.0),
        }
    }

    /// Clamp target to safe distance from anchor
    ///
    /// This is "The Leash" - prevents over-warping into Uncanny Valley.
    /// Returns NavigationWaypoint with clamping information.
    pub fn clamp_target(
        &self,
        target: &Vector30D,
        anchor: &Vector30D,
        anchor_island: Option<String>,
    ) -> NavigationWaypoint {
        let distance = target.distance_to(anchor);

        if distance <= self.max_safe_warp {
            // Safe! No clamping needed
            NavigationWaypoint {
                target: *target,
                mode: if distance < self.max_safe_warp * 0.5 {
                    NavigationMode::Interpolation
                } else {
                    NavigationMode::Extrapolation
                },
                anchor_island,
                distance_to_anchor: distance,
                was_clamped: false,
                original_target: None,
            }
        } else {
            // Too far! Apply clamping
            let direction = target.sub(anchor);
            let normalized_direction = direction.normalized();
            let safe_target = anchor.add(&normalized_direction.scale(self.max_safe_warp));

            warn!(
                "Clamping target: distance {} exceeds max safe warp {}",
                distance, self.max_safe_warp
            );

            NavigationWaypoint {
                target: safe_target,
                mode: NavigationMode::ExtrapolationClamped,
                anchor_island,
                distance_to_anchor: self.max_safe_warp,
                was_clamped: true,
                original_target: Some(*target),
            }
        }
    }

    /// Get max safe warp distance
    pub fn max_safe_warp(&self) -> f32 {
        self.max_safe_warp
    }
}

impl Default for SafetyClamp {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Nearest Neighbor Lookup (Priority 3: Critical)
// ============================================================================

/// Audio island in the navigation space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioIsland {
    /// Unique identifier
    pub key: String,
    /// 30D feature vector
    pub features: Vector30D,
    /// Audio samples (optional, may be loaded separately)
    pub audio: Option<Vec<f32>>,
    /// Species identifier
    pub species: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Phrase database with spatial indexing for fast nearest neighbor lookup
#[derive(Debug, Clone)]
pub struct PhraseDatabase {
    /// All islands indexed by key
    islands: HashMap<String, AudioIsland>,
    /// Islands indexed by species
    species_index: HashMap<String, Vec<String>>,
}

impl PhraseDatabase {
    /// Create a new empty phrase database
    pub fn new() -> Self {
        Self {
            islands: HashMap::new(),
            species_index: HashMap::new(),
        }
    }

    /// Add an island to the database
    pub fn add_island(&mut self, island: AudioIsland) {
        let key = island.key.clone();
        let species = island.species.clone();

        // Add to main index
        self.islands.insert(key.clone(), island);

        // Add to species index
        self.species_index.entry(species).or_default().push(key);
    }

    /// Find the nearest island to a target vector
    ///
    /// Returns None if database is empty.
    /// O(n) linear search - adequate for <10k phrases, can upgrade to KD-tree later.
    pub fn find_nearest_30d(&self, target: &Vector30D) -> Option<&AudioIsland> {
        if self.islands.is_empty() {
            return None;
        }

        let mut nearest_key: Option<&String> = None;
        let mut nearest_distance = f32::MAX;

        for (key, island) in &self.islands {
            let distance = target.distance_to(&island.features);
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_key = Some(key);
            }
        }

        nearest_key.and_then(|k| self.islands.get(k))
    }

    /// Find the nearest island within a specific species
    pub fn find_nearest_30d_species(
        &self,
        target: &Vector30D,
        species: &str,
    ) -> Option<&AudioIsland> {
        let keys = self.species_index.get(species)?;
        if keys.is_empty() {
            return None;
        }

        let mut nearest_key: Option<&String> = None;
        let mut nearest_distance = f32::MAX;

        for key in keys {
            if let Some(island) = self.islands.get(key) {
                let distance = target.distance_to(&island.features);
                if distance < nearest_distance {
                    nearest_distance = distance;
                    nearest_key = Some(key);
                }
            }
        }

        nearest_key.and_then(|k| self.islands.get(k))
    }

    /// Find k nearest neighbors
    pub fn find_k_nearest_30d(&self, target: &Vector30D, k: usize) -> Vec<&AudioIsland> {
        if self.islands.is_empty() {
            return Vec::new();
        }

        let mut distances: Vec<(&String, f32)> = self
            .islands
            .iter()
            .map(|(key, island)| (key, target.distance_to(&island.features)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .iter()
            .take(k)
            .filter_map(|(key, _)| self.islands.get(*key))
            .collect()
    }

    /// Get number of islands in database
    pub fn len(&self) -> usize {
        self.islands.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.islands.is_empty()
    }

    /// Get all islands for a species
    pub fn get_species_islands(&self, species: &str) -> Vec<&AudioIsland> {
        self.species_index
            .get(species)
            .map(|keys| keys.iter().filter_map(|k| self.islands.get(k)).collect())
            .unwrap_or_default()
    }
}

impl Default for PhraseDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Timeline Orchestration (Priority 4: High Impact)
// ============================================================================

/// Timeline event for real-time audio rendering
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineEvent {
    /// Start time in milliseconds
    pub start_ms: f32,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Audio buffer key
    pub buffer_key: String,
    /// Gain/amplitude (0.0 to 1.0)
    pub gain: f32,
    /// Modality/layer
    pub modality: String,
    /// Crossfade in duration (ms)
    pub crossfade_in_ms: f32,
    /// Crossfade out duration (ms)
    pub crossfade_out_ms: f32,
}

/// Timeline executor for real-time audio rendering
#[derive(Debug, Clone)]
pub struct TimelineExecutor {
    /// Sample rate in Hz
    sample_rate: usize,
}

impl TimelineExecutor {
    /// Create a new timeline executor
    pub fn new(sample_rate: usize) -> Self {
        Self { sample_rate }
    }

    /// Execute timeline with sample-accurate timing
    ///
    /// This is the REAL-TIME orchestration engine.
    /// Must complete in <100ms for typical 2-second timelines.
    pub fn execute_timeline(
        &self,
        events: &[TimelineEvent],
        source_buffers: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if events.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate total duration
        let max_end_time = events
            .iter()
            .map(|e| e.start_ms + e.duration_ms)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let total_samples = (max_end_time / 1000.0 * self.sample_rate as f32) as usize;
        let mut output = vec![0.0_f32; total_samples];

        // Render each event
        for event in events {
            if let Some(source) = source_buffers.get(&event.buffer_key) {
                self.render_event(&mut output, event, source)?;
            } else {
                warn!("Buffer not found: {}", event.buffer_key);
            }
        }

        Ok(output)
    }

    /// Render a single event to output buffer
    fn render_event(
        &self,
        output: &mut [f32],
        event: &TimelineEvent,
        source: &[f32],
    ) -> Result<()> {
        let start_sample = (event.start_ms / 1000.0 * self.sample_rate as f32) as usize;
        let duration_samples = (event.duration_ms / 1000.0 * self.sample_rate as f32) as usize;
        let end_sample = (start_sample + duration_samples).min(output.len());

        // Apply crossfade in
        let crossfade_in_samples =
            (event.crossfade_in_ms / 1000.0 * self.sample_rate as f32) as usize;
        let crossfade_out_samples =
            (event.crossfade_out_ms / 1000.0 * self.sample_rate as f32) as usize;

        for i in start_sample..end_sample {
            if i >= output.len() {
                break;
            }

            let source_idx = (i - start_sample) % source.len();
            let mut sample = source[source_idx] * event.gain;

            // Apply crossfade in
            if i < start_sample + crossfade_in_samples && crossfade_in_samples > 0 {
                let fade_pos = (i - start_sample) as f32 / crossfade_in_samples as f32;
                sample *= fade_pos;
            }

            // Apply crossfade out
            if i > end_sample - crossfade_out_samples && crossfade_out_samples > 0 {
                let fade_pos = (end_sample - i) as f32 / crossfade_out_samples as f32;
                sample *= fade_pos;
            }

            output[i] += sample;
        }

        Ok(())
    }
}

// ============================================================================
// Granular Delta Application (Priority 5: High Impact)
// ============================================================================

/// Granular synthesizer parameters
#[derive(Debug, Clone)]
pub struct GranularParams {
    /// Pitch shift ratio (1.0 = no shift)
    pub pitch_shift_ratio: f32,
    /// Grain size in milliseconds
    pub grain_size_ms: f32,
    /// Roughness amount (0.0 to 1.0)
    pub roughness_amount: f32,
    /// Duration scaling (1.0 = natural)
    pub duration_scale: f32,
}

impl Default for GranularParams {
    fn default() -> Self {
        Self {
            pitch_shift_ratio: 1.0,
            grain_size_ms: 20.0,
            roughness_amount: 0.0,
            duration_scale: 1.0,
        }
    }
}

/// Apply 30D delta to granular synthesis parameters
///
/// This maps the 30D acoustic delta to synthesizer control parameters.
/// This is the PRIMARY integration point for Acoustic Algebra → Rust Synthesis.
pub fn apply_delta_to_granular(
    delta: &VectorDelta,
    base_params: &GranularParams,
    source_metadata: &Vector30D,
) -> GranularParams {
    // Map delta_mean_f0_hz to pitch shift ratio
    // Formula: ratio = 1 + (delta_f0 / source_f0)
    let pitch_shift_ratio = if source_metadata.mean_f0_hz > 0.0 {
        1.0 + (delta.delta_mean_f0_hz / source_metadata.mean_f0_hz)
    } else {
        1.0
    };

    // Map delta_duration_ms to duration scale
    let duration_scale = if source_metadata.duration_ms > 0.0 {
        1.0 + (delta.delta_duration_ms / source_metadata.duration_ms)
    } else {
        1.0
    };

    // Map spectral_flatness to roughness (higher flatness = more roughness)
    let roughness_amount =
        (base_params.roughness_amount + delta.delta_spectral_flatness).clamp(0.0, 1.0);

    // Map jitter to grain size variation (not directly modifying grain_size_ms here,
    // but this could be used for granular jitter effects)
    let _jitter_effect = delta.delta_jitter;

    GranularParams {
        pitch_shift_ratio,
        grain_size_ms: base_params.grain_size_ms,
        roughness_amount,
        duration_scale,
    }
}

// ============================================================================
// Navigation Engine (High-Level API)
// ============================================================================

/// Navigation engine for Island Hopping
#[derive(Debug, Clone)]
pub struct NavigationEngine {
    /// Safety clamp for limiting warp distance
    clamp: SafetyClamp,
    /// Phrase database for nearest neighbor lookup
    database: PhraseDatabase,
}

impl NavigationEngine {
    /// Create a new navigation engine
    pub fn new() -> Self {
        Self {
            clamp: SafetyClamp::new(),
            database: PhraseDatabase::new(),
        }
    }

    /// Create with custom max warp distance
    pub fn with_max_warp(max_safe_warp: f32) -> Self {
        Self {
            clamp: SafetyClamp::with_max_warp(max_safe_warp),
            database: PhraseDatabase::new(),
        }
    }

    /// Interpolate between two vectors (Bridge Builder - SAFE)
    pub fn interpolate(&self, start: &Vector30D, end: &Vector30D, alpha: f32) -> Vector30D {
        start.interpolate(end, alpha)
    }

    /// Extrapolate from origin in direction (Ocean Explorer - RISKY)
    pub fn extrapolate(
        &self,
        origin: &Vector30D,
        direction: &VectorDelta,
        factor: f32,
    ) -> Vector30D {
        origin.extrapolate(direction, factor)
    }

    /// Apply safety clamping to target
    pub fn clamp_to_safe_distance(
        &self,
        target: &Vector30D,
        anchor: &Vector30D,
        anchor_island: Option<String>,
    ) -> NavigationWaypoint {
        self.clamp.clamp_target(target, anchor, anchor_island)
    }

    /// Find nearest island to target vector
    pub fn find_nearest_island(&self, target: &Vector30D) -> Option<&AudioIsland> {
        self.database.find_nearest_30d(target)
    }

    /// Add an island to the database
    pub fn add_island(&mut self, island: AudioIsland) {
        self.database.add_island(island);
    }

    /// Get reference to phrase database
    pub fn database(&self) -> &PhraseDatabase {
        &self.database
    }

    /// Get mutable reference to phrase database
    pub fn database_mut(&mut self) -> &mut PhraseDatabase {
        &mut self.database
    }
}

impl Default for NavigationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test helpers
    fn create_test_vector(f0: f32, duration: f32) -> Vector30D {
        Vector30D {
            mean_f0_hz: f0,
            duration_ms: duration,
            ..Default::default()
        }
    }

    fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() < epsilon,
            "Values not approximately equal: {} vs {} (epsilon: {})",
            a,
            b,
            epsilon
        );
    }

    // =========================================================================
    // Vector30D Tests
    // =========================================================================

    #[test]
    fn test_vector30d_default() {
        let v = Vector30D::default();
        assert_approx_eq(v.mean_f0_hz, 7000.0, 1e-5);
        assert_approx_eq(v.duration_ms, 50.0, 1e-5);
        assert_approx_eq(v.harmonicity, 0.8, 1e-5);
        assert_approx_eq(v.shimmer, 0.03, 1e-5);
        assert_approx_eq(v.mfcc_13, 0.4, 1e-5);
        assert_approx_eq(v.spectral_flux, 0.5, 1e-5);
    }

    #[test]
    fn test_vector30d_new() {
        let v = Vector30D::new(
            8000.0, 500.0, 60.0, 25.0, 0.4, 0.9, 10.0, 25.0, 0.8, 8.0, 0.03, 0.02, 0.05, -12.0,
            -6.0, -3.0, -1.5, -0.8, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.6, 180.0, 10.0,
            0.4,
        );
        assert_approx_eq(v.mean_f0_hz, 8000.0, 1e-5);
        assert_approx_eq(v.duration_ms, 60.0, 1e-5);
        assert_approx_eq(v.f0_range_hz, 500.0, 1e-5);
        assert_approx_eq(v.harmonicity, 0.9, 1e-5);
        assert_approx_eq(v.shimmer, 0.05, 1e-5);
        assert_approx_eq(v.mfcc_13, 0.3, 1e-5);
        assert_approx_eq(v.spectral_flux, 0.6, 1e-5);
    }

    #[test]
    fn test_distance_to_self() {
        let v1 = create_test_vector(7000.0, 50.0);
        assert_approx_eq(v1.distance_to(&v1), 0.0, 1e-5);
    }

    #[test]
    fn test_distance_to_different() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 50.0);
        let distance = v1.distance_to(&v2);

        // Distance should be positive
        assert!(distance > 0.0);

        // With 1000 Hz difference and 2000 Hz range, normalized diff is 0.5
        // sqrt(0.5^2) = 0.5
        assert_approx_eq(distance, 0.5, 0.01);
    }

    #[test]
    fn test_interpolate_alpha_zero() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 60.0);
        let result = v1.interpolate(&v2, 0.0);

        assert_approx_eq(result.mean_f0_hz, v1.mean_f0_hz, 1e-5);
        assert_approx_eq(result.duration_ms, v1.duration_ms, 1e-5);
    }

    #[test]
    fn test_interpolate_alpha_one() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 60.0);
        let result = v1.interpolate(&v2, 1.0);

        assert_approx_eq(result.mean_f0_hz, v2.mean_f0_hz, 1e-5);
        assert_approx_eq(result.duration_ms, v2.duration_ms, 1e-5);
    }

    #[test]
    fn test_interpolate_alpha_half() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 60.0);
        let result = v1.interpolate(&v2, 0.5);

        assert_approx_eq(result.mean_f0_hz, 7500.0, 1e-5);
        assert_approx_eq(result.duration_ms, 55.0, 1e-5);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in [0, 1]")]
    fn test_interpolate_invalid_alpha() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 60.0);
        v1.interpolate(&v2, 1.5);
    }

    #[test]
    fn test_extrapolate_zero_factor() {
        let v1 = create_test_vector(7000.0, 50.0);
        let mut delta = VectorDelta::zero();
        delta.delta_mean_f0_hz = 1000.0;
        delta.delta_duration_ms = 10.0;
        let result = v1.extrapolate(&delta, 0.0);

        assert_approx_eq(result.mean_f0_hz, 7000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 50.0, 1e-5);
    }

    #[test]
    fn test_extrapolate_unit_factor() {
        let v1 = create_test_vector(7000.0, 50.0);
        let mut delta = VectorDelta::zero();
        delta.delta_mean_f0_hz = 1000.0;
        delta.delta_duration_ms = 10.0;
        let result = v1.extrapolate(&delta, 1.0);

        assert_approx_eq(result.mean_f0_hz, 8000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 60.0, 1e-5);
    }

    #[test]
    fn test_extrapolate_double_factor() {
        let v1 = create_test_vector(7000.0, 50.0);
        let mut delta = VectorDelta::zero();
        delta.delta_mean_f0_hz = 1000.0;
        delta.delta_duration_ms = 10.0;
        let result = v1.extrapolate(&delta, 2.0);

        assert_approx_eq(result.mean_f0_hz, 9000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 70.0, 1e-5);
    }

    #[test]
    fn test_vector_add() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(1000.0, 10.0);
        let result = v1.add(&v2);

        assert_approx_eq(result.mean_f0_hz, 8000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 60.0, 1e-5);
    }

    #[test]
    fn test_vector_sub() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(1000.0, 10.0);
        let result = v1.sub(&v2);

        assert_approx_eq(result.mean_f0_hz, 6000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 40.0, 1e-5);
    }

    #[test]
    fn test_vector_scale() {
        let v1 = create_test_vector(7000.0, 50.0);
        let result = v1.scale(2.0);

        assert_approx_eq(result.mean_f0_hz, 14000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 100.0, 1e-5);
    }

    #[test]
    fn test_vector_magnitude() {
        let v1 = create_test_vector(7000.0, 50.0);
        let mag = v1.magnitude();

        // Magnitude should be positive
        assert!(mag > 0.0);
    }

    #[test]
    fn test_vector_normalized() {
        let v1 = create_test_vector(7000.0, 50.0);
        let normalized = v1.normalized();
        let mag = normalized.magnitude();

        // Normalized vector should have magnitude ~1.0
        assert_approx_eq(mag, 1.0, 0.01);
    }

    // =========================================================================
    // VectorDelta Tests
    // =========================================================================

    #[test]
    fn test_delta_zero() {
        let delta = VectorDelta::zero();
        assert_approx_eq(delta.delta_mean_f0_hz, 0.0, 1e-5);
        assert_approx_eq(delta.delta_duration_ms, 0.0, 1e-5);
        assert_approx_eq(delta.delta_harmonicity, 0.0, 1e-5);
        assert_approx_eq(delta.delta_shimmer, 0.0, 1e-5);
        assert_approx_eq(delta.delta_mfcc_13, 0.0, 1e-5);
        assert_approx_eq(delta.delta_spectral_flux, 0.0, 1e-5);
    }

    #[test]
    fn test_delta_from_vectors() {
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 60.0);
        let delta = VectorDelta::from_vectors(&v2, &v1);

        assert_approx_eq(delta.delta_mean_f0_hz, 1000.0, 1e-5);
        assert_approx_eq(delta.delta_duration_ms, 10.0, 1e-5);
    }

    // =========================================================================
    // SafetyClamp Tests
    // =========================================================================

    #[test]
    fn test_clamp_safe_distance() {
        let clamp = SafetyClamp::with_max_warp(0.5);

        let anchor = create_test_vector(7000.0, 50.0);
        let target = create_test_vector(7100.0, 50.5); // Very small distance (< 0.25 * max_warp)

        let result = clamp.clamp_target(&target, &anchor, Some("island1".to_string()));

        assert!(!result.was_clamped);
        // Distance is ~0.05 which is < 0.25 (0.5 * 0.5), so should be Interpolation
        assert_eq!(result.mode, NavigationMode::Interpolation);
    }

    #[test]
    fn test_clamp_unsafe_distance() {
        let clamp = SafetyClamp::with_max_warp(0.2);

        let anchor = create_test_vector(7000.0, 50.0);
        let target = create_test_vector(9000.0, 70.0); // Large distance

        let result = clamp.clamp_target(&target, &anchor, Some("island1".to_string()));

        assert!(result.was_clamped);
        assert_eq!(result.mode, NavigationMode::ExtrapolationClamped);
        assert!(result.original_target.is_some());
    }

    #[test]
    fn test_clamp_max_distance() {
        let clamp = SafetyClamp::with_max_warp(0.3);

        let anchor = create_test_vector(7000.0, 50.0);
        let target = create_test_vector(8000.0, 60.0); // Medium distance

        let result = clamp.clamp_target(&target, &anchor, Some("island1".to_string()));

        // After clamping, distance should equal max_safe_warp
        assert_approx_eq(result.distance_to_anchor, 0.3, 0.01);
    }

    // =========================================================================
    // PhraseDatabase Tests
    // =========================================================================

    #[test]
    fn test_database_empty() {
        let db = PhraseDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_database_add_island() {
        let mut db = PhraseDatabase::new();
        let island = AudioIsland {
            key: "island1".to_string(),
            features: create_test_vector(7000.0, 50.0),
            audio: None,
            species: "marmoset".to_string(),
            metadata: HashMap::new(),
        };

        db.add_island(island);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_database_find_nearest() {
        let mut db = PhraseDatabase::new();

        db.add_island(AudioIsland {
            key: "island1".to_string(),
            features: create_test_vector(7000.0, 50.0),
            audio: None,
            species: "marmoset".to_string(),
            metadata: HashMap::new(),
        });

        db.add_island(AudioIsland {
            key: "island2".to_string(),
            features: create_test_vector(8000.0, 60.0),
            audio: None,
            species: "marmoset".to_string(),
            metadata: HashMap::new(),
        });

        let target = create_test_vector(7500.0, 55.0);
        let nearest = db.find_nearest_30d(&target);

        assert!(nearest.is_some());
        // Should find island1 or island2 (both are close)
        assert!(nearest.unwrap().key == "island1" || nearest.unwrap().key == "island2");
    }

    #[test]
    fn test_database_find_nearest_species() {
        let mut db = PhraseDatabase::new();

        db.add_island(AudioIsland {
            key: "marmoset1".to_string(),
            features: create_test_vector(7000.0, 50.0),
            audio: None,
            species: "marmoset".to_string(),
            metadata: HashMap::new(),
        });

        db.add_island(AudioIsland {
            key: "bat1".to_string(),
            features: create_test_vector(7080.0, 50.8), // Clearly closer to target (7100, 51)
            audio: None,
            species: "bat".to_string(),
            metadata: HashMap::new(),
        });

        let target = create_test_vector(7100.0, 51.0);

        // Nearest overall should be bat1 (distance ~0.01 vs marmoset1 distance ~0.054)
        let nearest = db.find_nearest_30d(&target);
        assert_eq!(nearest.unwrap().key, "bat1");

        // Nearest marmoset should be marmoset1
        let nearest_marmoset = db.find_nearest_30d_species(&target, "marmoset");
        assert_eq!(nearest_marmoset.unwrap().key, "marmoset1");
    }

    #[test]
    fn test_database_find_k_nearest() {
        let mut db = PhraseDatabase::new();

        for i in 0..10 {
            db.add_island(AudioIsland {
                key: format!("island{}", i),
                features: create_test_vector(7000.0 + i as f32 * 100.0, 50.0),
                audio: None,
                species: "marmoset".to_string(),
                metadata: HashMap::new(),
            });
        }

        let target = create_test_vector(7250.0, 50.0);
        let k_nearest = db.find_k_nearest_30d(&target, 3);

        assert_eq!(k_nearest.len(), 3);
    }

    // =========================================================================
    // NavigationEngine Tests
    // =========================================================================

    #[test]
    fn test_navigation_engine_interpolate() {
        let engine = NavigationEngine::new();
        let v1 = create_test_vector(7000.0, 50.0);
        let v2 = create_test_vector(8000.0, 60.0);
        let result = engine.interpolate(&v1, &v2, 0.5);

        assert_approx_eq(result.mean_f0_hz, 7500.0, 1e-5);
    }

    #[test]
    fn test_navigation_engine_clamp() {
        let engine = NavigationEngine::with_max_warp(0.2);

        let anchor = create_test_vector(7000.0, 50.0);
        let target = create_test_vector(9000.0, 70.0); // Far away

        let result = engine.clamp_to_safe_distance(&target, &anchor, Some("island1".to_string()));

        assert!(result.was_clamped);
        assert_approx_eq(result.distance_to_anchor, 0.2, 0.01);
    }

    #[test]
    fn test_navigation_engine_find_nearest() {
        let mut engine = NavigationEngine::new();

        engine.add_island(AudioIsland {
            key: "island1".to_string(),
            features: create_test_vector(7000.0, 50.0),
            audio: None,
            species: "marmoset".to_string(),
            metadata: HashMap::new(),
        });

        engine.add_island(AudioIsland {
            key: "island2".to_string(),
            features: create_test_vector(8000.0, 60.0),
            audio: None,
            species: "marmoset".to_string(),
            metadata: HashMap::new(),
        });

        let target = create_test_vector(7500.0, 55.0);
        let nearest = engine.find_nearest_island(&target);

        assert!(nearest.is_some());
    }

    // =========================================================================
    // Granular Delta Application Tests
    // =========================================================================

    #[test]
    fn test_apply_delta_to_granular() {
        let mut delta = VectorDelta::zero();
        delta.delta_mean_f0_hz = 1000.0;
        delta.delta_duration_ms = 10.0;
        delta.delta_spectral_flatness = 0.1;
        delta.delta_jitter = 0.01;

        let base_params = GranularParams::default();
        let source_metadata = create_test_vector(7000.0, 50.0);

        let result = apply_delta_to_granular(&delta, &base_params, &source_metadata);

        // Pitch shift ratio should be ~1.14 (1000/7000 + 1)
        assert_approx_eq(result.pitch_shift_ratio, 1.0 + (1000.0 / 7000.0), 0.01);

        // Duration scale should be ~1.2 (10/50 + 1)
        assert_approx_eq(result.duration_scale, 1.0 + (10.0 / 50.0), 0.01);

        // Roughness should increase by 0.1
        assert_approx_eq(result.roughness_amount, 0.1, 0.01);
    }

    // =========================================================================
    // Timeline Executor Tests
    // =========================================================================

    #[test]
    fn test_timeline_executor_empty() {
        let executor = TimelineExecutor::new(48000);
        let events = [];
        let buffers = HashMap::new();

        let result = executor.execute_timeline(&events, &buffers);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_timeline_executor_single_event() {
        let executor = TimelineExecutor::new(48000);

        let mut source = vec![0.0_f32; 2400]; // 50ms at 48kHz
        for i in 0..source.len() {
            source[i] = ((i as f32) / 2400.0) * 2.0 - 1.0; // -1 to 1 ramp
        }

        let mut buffers = HashMap::new();
        buffers.insert("buffer1".to_string(), source);

        let events = vec![TimelineEvent {
            start_ms: 0.0,
            duration_ms: 50.0,
            buffer_key: "buffer1".to_string(),
            gain: 1.0,
            modality: "test".to_string(),
            crossfade_in_ms: 0.0,
            crossfade_out_ms: 0.0,
        }];

        let result = executor.execute_timeline(&events, &buffers);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 2400); // 50ms at 48kHz
    }

    #[test]
    fn test_timeline_executor_crossfade() {
        let executor = TimelineExecutor::new(48000);

        let source1 = vec![1.0_f32; 2400]; // 50ms
        let source2 = vec![1.0_f32; 2400]; // 50ms

        let mut buffers = HashMap::new();
        buffers.insert("buffer1".to_string(), source1);
        buffers.insert("buffer2".to_string(), source2);

        let events = vec![
            TimelineEvent {
                start_ms: 0.0,
                duration_ms: 50.0,
                buffer_key: "buffer1".to_string(),
                gain: 1.0,
                modality: "test".to_string(),
                crossfade_in_ms: 10.0,
                crossfade_out_ms: 0.0,
            },
            TimelineEvent {
                start_ms: 50.0,
                duration_ms: 50.0,
                buffer_key: "buffer2".to_string(),
                gain: 1.0,
                modality: "test".to_string(),
                crossfade_in_ms: 0.0,
                crossfade_out_ms: 10.0,
            },
        ];

        let result = executor.execute_timeline(&events, &buffers);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 4800); // 100ms at 48kHz
    }
}
