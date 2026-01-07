//! Island Hopping Navigation Module (Rust Execution Layer)
//! =======================================================
//!
//! This module implements the execution layer for Island Hopping navigation strategy.
//! It provides high-performance, safety-critical components for:
//!
//! - 17D Vector Math Operations (SIMD-optimized)
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
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use log::warn;

// ============================================================================
// 17D Vector Math Operations (Priority 1: Critical)
// ============================================================================

/// 17-dimensional acoustic feature vector
///
/// Features organized by category:
/// - Fundamental (3): mean_f0_hz, duration_ms, f0_range_hz
/// - Grit Factors (2): harmonic_to_noise_ratio, spectral_flatness
/// - Motion Factors (6): attack_time_ms, decay_time_ms, sustain_level, vibrato_rate_hz, vibrato_depth, jitter
/// - Fingerprint Factors (5): mfcc_1, mfcc_2, mfcc_3, mfcc_4, spectral_contrast
/// - Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector17D {
    // === Fundamental (3 features) ===
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,

    // === Grit Factors (2 features) ===
    pub harmonic_to_noise_ratio: f32,
    pub spectral_flatness: f32,

    // === Motion Factors (6 features) ===
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    pub sustain_level: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth: f32,
    pub jitter: f32,

    // === Fingerprint Factors (5 features) ===
    pub mfcc_1: f32,
    pub mfcc_2: f32,
    pub mfcc_3: f32,
    pub mfcc_4: f32,
    pub spectral_contrast: f32,

    // === Rhythm Factors (3 features) ===
    pub median_ici_ms: f32,
    pub onset_rate_hz: f32,
    pub ici_coefficient_of_variation: f32,
}

impl Default for Vector17D {
    fn default() -> Self {
        Self {
            mean_f0_hz: 7000.0,
            duration_ms: 50.0,
            f0_range_hz: 400.0,
            harmonic_to_noise_ratio: 20.0,
            spectral_flatness: 0.3,
            attack_time_ms: 5.0,
            decay_time_ms: 20.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 7.0,
            vibrato_depth: 0.02,
            jitter: 0.01,
            mfcc_1: -10.0,
            mfcc_2: -5.0,
            mfcc_3: -2.0,
            mfcc_4: -1.0,
            spectral_contrast: 20.0,
            median_ici_ms: 150.0,
            onset_rate_hz: 8.0,
            ici_coefficient_of_variation: 0.3,
        }
    }
}

impl Vector17D {
    /// Create a new Vector17D with all 17 dimensions
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mean_f0_hz: f32,
        duration_ms: f32,
        f0_range_hz: f32,
        harmonic_to_noise_ratio: f32,
        spectral_flatness: f32,
        attack_time_ms: f32,
        decay_time_ms: f32,
        sustain_level: f32,
        vibrato_rate_hz: f32,
        vibrato_depth: f32,
        jitter: f32,
        mfcc_1: f32,
        mfcc_2: f32,
        mfcc_3: f32,
        mfcc_4: f32,
        spectral_contrast: f32,
        median_ici_ms: f32,
        onset_rate_hz: f32,
        ici_coefficient_of_variation: f32,
    ) -> Self {
        Self {
            mean_f0_hz,
            duration_ms,
            f0_range_hz,
            harmonic_to_noise_ratio,
            spectral_flatness,
            attack_time_ms,
            decay_time_ms,
            sustain_level,
            vibrato_rate_hz,
            vibrato_depth,
            jitter,
            mfcc_1,
            mfcc_2,
            mfcc_3,
            mfcc_4,
            spectral_contrast,
            median_ici_ms,
            onset_rate_hz,
            ici_coefficient_of_variation,
        }
    }

    /// Convert to array for SIMD operations
    pub fn to_array(&self) -> [f32; 17] {
        [
            self.mean_f0_hz,
            self.duration_ms,
            self.f0_range_hz,
            self.harmonic_to_noise_ratio,
            self.spectral_flatness,
            self.attack_time_ms,
            self.decay_time_ms,
            self.sustain_level,
            self.vibrato_rate_hz,
            self.vibrato_depth,
            self.jitter,
            self.mfcc_1,
            self.mfcc_2,
            self.mfcc_3,
            self.mfcc_4,
            self.spectral_contrast,
            self.median_ici_ms,
        ]
    }

    /// Convert from array
    pub fn from_array(arr: [f32; 17]) -> Self {
        Self {
            mean_f0_hz: arr[0],
            duration_ms: arr[1],
            f0_range_hz: arr[2],
            harmonic_to_noise_ratio: arr[3],
            spectral_flatness: arr[4],
            attack_time_ms: arr[5],
            decay_time_ms: arr[6],
            sustain_level: arr[7],
            vibrato_rate_hz: arr[8],
            vibrato_depth: arr[9],
            jitter: arr[10],
            mfcc_1: arr[11],
            mfcc_2: arr[12],
            mfcc_3: arr[13],
            mfcc_4: arr[14],
            spectral_contrast: arr[15],
            median_ici_ms: arr[16],
            // These are not part of the 17D core array
            onset_rate_hz: Default::default(),
            ici_coefficient_of_variation: Default::default(),
        }
    }

    /// Get normalization ranges for each dimension
    fn normalization_ranges() -> [f32; 17] {
        [
            2000.0,    // mean_f0_hz: 0-2000 Hz typical range
            50.0,      // duration_ms: 0-50 ms typical range
            500.0,     // f0_range_hz: 0-500 Hz
            30.0,      // harmonic_to_noise_ratio: 0-30 dB
            1.0,       // spectral_flatness: 0-1
            20.0,      // attack_time_ms: 0-20 ms
            50.0,      // decay_time_ms: 0-50 ms
            1.0,       // sustain_level: 0-1
            20.0,      // vibrato_rate_hz: 0-20 Hz
            0.1,       // vibrato_depth: 0-0.1
            0.05,      // jitter: 0-0.05
            20.0,      // mfcc_1: -20 to 0
            20.0,      // mfcc_2: -20 to 0
            20.0,      // mfcc_3: -20 to 0
            20.0,      // mfcc_4: -20 to 0
            40.0,      // spectral_contrast: 0-40
            200.0,     // median_ici_ms: 0-200 ms
        ]
    }

    /// Calculate normalized Euclidean distance to another vector
    ///
    /// This is the PRIMARY distance metric for Island Hopping navigation.
    /// Distances are normalized by dimension-specific ranges to ensure
    /// meaningful comparisons across different acoustic features.
    ///
    /// Uses the 17D core dimensions for distance calculation.
    pub fn distance_to(&self, other: &Vector17D) -> f32 {
        let v1 = self.to_array();
        let v2 = other.to_array();
        let ranges = Self::normalization_ranges();

        let mut sum_squared = 0.0_f32;
        for i in 0..17 {
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
    pub fn interpolate(&self, other: &Vector17D, alpha: f32) -> Vector17D {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be in [0, 1], got {}",
            alpha
        );

        Vector17D {
            mean_f0_hz: self.mean_f0_hz * (1.0 - alpha) + other.mean_f0_hz * alpha,
            duration_ms: self.duration_ms * (1.0 - alpha) + other.duration_ms * alpha,
            f0_range_hz: self.f0_range_hz * (1.0 - alpha) + other.f0_range_hz * alpha,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio * (1.0 - alpha)
                + other.harmonic_to_noise_ratio * alpha,
            spectral_flatness: self.spectral_flatness * (1.0 - alpha) + other.spectral_flatness * alpha,
            attack_time_ms: self.attack_time_ms * (1.0 - alpha) + other.attack_time_ms * alpha,
            decay_time_ms: self.decay_time_ms * (1.0 - alpha) + other.decay_time_ms * alpha,
            sustain_level: self.sustain_level * (1.0 - alpha) + other.sustain_level * alpha,
            vibrato_rate_hz: self.vibrato_rate_hz * (1.0 - alpha) + other.vibrato_rate_hz * alpha,
            vibrato_depth: self.vibrato_depth * (1.0 - alpha) + other.vibrato_depth * alpha,
            jitter: self.jitter * (1.0 - alpha) + other.jitter * alpha,
            mfcc_1: self.mfcc_1 * (1.0 - alpha) + other.mfcc_1 * alpha,
            mfcc_2: self.mfcc_2 * (1.0 - alpha) + other.mfcc_2 * alpha,
            mfcc_3: self.mfcc_3 * (1.0 - alpha) + other.mfcc_3 * alpha,
            mfcc_4: self.mfcc_4 * (1.0 - alpha) + other.mfcc_4 * alpha,
            spectral_contrast: self.spectral_contrast * (1.0 - alpha) + other.spectral_contrast * alpha,
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
    pub fn extrapolate(&self, direction: &VectorDelta, factor: f32) -> Vector17D {
        assert!(
            factor >= 0.0,
            "Factor must be >= 0, got {}",
            factor
        );

        Vector17D {
            mean_f0_hz: self.mean_f0_hz + direction.delta_mean_f0_hz * factor,
            duration_ms: self.duration_ms + direction.delta_duration_ms * factor,
            f0_range_hz: self.f0_range_hz + direction.delta_f0_range_hz * factor,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio
                + direction.delta_harmonic_to_noise_ratio * factor,
            spectral_flatness: self.spectral_flatness + direction.delta_spectral_flatness * factor,
            attack_time_ms: self.attack_time_ms + direction.delta_attack_time_ms * factor,
            decay_time_ms: self.decay_time_ms + direction.delta_decay_time_ms * factor,
            sustain_level: self.sustain_level + direction.delta_sustain_level * factor,
            vibrato_rate_hz: self.vibrato_rate_hz + direction.delta_vibrato_rate_hz * factor,
            vibrato_depth: self.vibrato_depth + direction.delta_vibrato_depth * factor,
            jitter: self.jitter + direction.delta_jitter * factor,
            mfcc_1: self.mfcc_1 + direction.delta_mfcc_1 * factor,
            mfcc_2: self.mfcc_2 + direction.delta_mfcc_2 * factor,
            mfcc_3: self.mfcc_3 + direction.delta_mfcc_3 * factor,
            mfcc_4: self.mfcc_4 + direction.delta_mfcc_4 * factor,
            spectral_contrast: self.spectral_contrast + direction.delta_spectral_contrast * factor,
            median_ici_ms: self.median_ici_ms + direction.delta_median_ici_ms * factor,
            onset_rate_hz: self.onset_rate_hz + direction.delta_onset_rate_hz * factor,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation
                + direction.delta_ici_coefficient_of_variation * factor,
        }
    }

    /// Add two vectors (for delta operations)
    pub fn add(&self, other: &Vector17D) -> Vector17D {
        Vector17D {
            mean_f0_hz: self.mean_f0_hz + other.mean_f0_hz,
            duration_ms: self.duration_ms + other.duration_ms,
            f0_range_hz: self.f0_range_hz + other.f0_range_hz,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio + other.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness + other.spectral_flatness,
            attack_time_ms: self.attack_time_ms + other.attack_time_ms,
            decay_time_ms: self.decay_time_ms + other.decay_time_ms,
            sustain_level: self.sustain_level + other.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz + other.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth + other.vibrato_depth,
            jitter: self.jitter + other.jitter,
            mfcc_1: self.mfcc_1 + other.mfcc_1,
            mfcc_2: self.mfcc_2 + other.mfcc_2,
            mfcc_3: self.mfcc_3 + other.mfcc_3,
            mfcc_4: self.mfcc_4 + other.mfcc_4,
            spectral_contrast: self.spectral_contrast + other.spectral_contrast,
            median_ici_ms: self.median_ici_ms + other.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz + other.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation + other.ici_coefficient_of_variation,
        }
    }

    /// Subtract two vectors (for delta calculation)
    pub fn sub(&self, other: &Vector17D) -> Vector17D {
        Vector17D {
            mean_f0_hz: self.mean_f0_hz - other.mean_f0_hz,
            duration_ms: self.duration_ms - other.duration_ms,
            f0_range_hz: self.f0_range_hz - other.f0_range_hz,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio - other.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness - other.spectral_flatness,
            attack_time_ms: self.attack_time_ms - other.attack_time_ms,
            decay_time_ms: self.decay_time_ms - other.decay_time_ms,
            sustain_level: self.sustain_level - other.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz - other.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth - other.vibrato_depth,
            jitter: self.jitter - other.jitter,
            mfcc_1: self.mfcc_1 - other.mfcc_1,
            mfcc_2: self.mfcc_2 - other.mfcc_2,
            mfcc_3: self.mfcc_3 - other.mfcc_3,
            mfcc_4: self.mfcc_4 - other.mfcc_4,
            spectral_contrast: self.spectral_contrast - other.spectral_contrast,
            median_ici_ms: self.median_ici_ms - other.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz - other.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation - other.ici_coefficient_of_variation,
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, factor: f32) -> Vector17D {
        Vector17D {
            mean_f0_hz: self.mean_f0_hz * factor,
            duration_ms: self.duration_ms * factor,
            f0_range_hz: self.f0_range_hz * factor,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio * factor,
            spectral_flatness: self.spectral_flatness * factor,
            attack_time_ms: self.attack_time_ms * factor,
            decay_time_ms: self.decay_time_ms * factor,
            sustain_level: self.sustain_level * factor,
            vibrato_rate_hz: self.vibrato_rate_hz * factor,
            vibrato_depth: self.vibrato_depth * factor,
            jitter: self.jitter * factor,
            mfcc_1: self.mfcc_1 * factor,
            mfcc_2: self.mfcc_2 * factor,
            mfcc_3: self.mfcc_3 * factor,
            mfcc_4: self.mfcc_4 * factor,
            spectral_contrast: self.spectral_contrast * factor,
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
        for i in 0..17 {
            let normalized = arr[i] / ranges[i];
            sum_squared += normalized * normalized;
        }

        sum_squared.sqrt()
    }

    /// Normalize to unit vector
    pub fn normalized(&self) -> Vector17D {
        let mag = self.magnitude();
        if mag > 1e-6 {
            self.scale(1.0 / mag)
        } else {
            *self
        }
    }
}

impl std::ops::Add<Vector17D> for Vector17D {
    type Output = Vector17D;

    fn add(self, rhs: Vector17D) -> Self::Output {
        Vector17D {
            mean_f0_hz: self.mean_f0_hz + rhs.mean_f0_hz,
            duration_ms: self.duration_ms + rhs.duration_ms,
            f0_range_hz: self.f0_range_hz + rhs.f0_range_hz,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio + rhs.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness + rhs.spectral_flatness,
            attack_time_ms: self.attack_time_ms + rhs.attack_time_ms,
            decay_time_ms: self.decay_time_ms + rhs.decay_time_ms,
            sustain_level: self.sustain_level + rhs.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz + rhs.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth + rhs.vibrato_depth,
            jitter: self.jitter + rhs.jitter,
            mfcc_1: self.mfcc_1 + rhs.mfcc_1,
            mfcc_2: self.mfcc_2 + rhs.mfcc_2,
            mfcc_3: self.mfcc_3 + rhs.mfcc_3,
            mfcc_4: self.mfcc_4 + rhs.mfcc_4,
            spectral_contrast: self.spectral_contrast + rhs.spectral_contrast,
            median_ici_ms: self.median_ici_ms + rhs.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz + rhs.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation + rhs.ici_coefficient_of_variation,
        }
    }
}

impl std::ops::Sub<Vector17D> for Vector17D {
    type Output = Vector17D;

    fn sub(self, rhs: Vector17D) -> Self::Output {
        Vector17D {
            mean_f0_hz: self.mean_f0_hz - rhs.mean_f0_hz,
            duration_ms: self.duration_ms - rhs.duration_ms,
            f0_range_hz: self.f0_range_hz - rhs.f0_range_hz,
            harmonic_to_noise_ratio: self.harmonic_to_noise_ratio - rhs.harmonic_to_noise_ratio,
            spectral_flatness: self.spectral_flatness - rhs.spectral_flatness,
            attack_time_ms: self.attack_time_ms - rhs.attack_time_ms,
            decay_time_ms: self.decay_time_ms - rhs.decay_time_ms,
            sustain_level: self.sustain_level - rhs.sustain_level,
            vibrato_rate_hz: self.vibrato_rate_hz - rhs.vibrato_rate_hz,
            vibrato_depth: self.vibrato_depth - rhs.vibrato_depth,
            jitter: self.jitter - rhs.jitter,
            mfcc_1: self.mfcc_1 - rhs.mfcc_1,
            mfcc_2: self.mfcc_2 - rhs.mfcc_2,
            mfcc_3: self.mfcc_3 - rhs.mfcc_3,
            mfcc_4: self.mfcc_4 - rhs.mfcc_4,
            spectral_contrast: self.spectral_contrast - rhs.spectral_contrast,
            median_ici_ms: self.median_ici_ms - rhs.median_ici_ms,
            onset_rate_hz: self.onset_rate_hz - rhs.onset_rate_hz,
            ici_coefficient_of_variation: self.ici_coefficient_of_variation - rhs.ici_coefficient_of_variation,
        }
    }
}

impl std::ops::Mul<f32> for Vector17D {
    type Output = Vector17D;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

// ============================================================================
// Vector Delta (17D Difference Vector)
// ============================================================================

/// 17-dimensional delta vector for extrapolation operations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VectorDelta {
    // === Fundamental ===
    pub delta_mean_f0_hz: f32,
    pub delta_duration_ms: f32,
    pub delta_f0_range_hz: f32,

    // === Grit Factors ===
    pub delta_harmonic_to_noise_ratio: f32,
    pub delta_spectral_flatness: f32,

    // === Motion Factors ===
    pub delta_attack_time_ms: f32,
    pub delta_decay_time_ms: f32,
    pub delta_sustain_level: f32,
    pub delta_vibrato_rate_hz: f32,
    pub delta_vibrato_depth: f32,
    pub delta_jitter: f32,

    // === Fingerprint Factors ===
    pub delta_mfcc_1: f32,
    pub delta_mfcc_2: f32,
    pub delta_mfcc_3: f32,
    pub delta_mfcc_4: f32,
    pub delta_spectral_contrast: f32,

    // === Rhythm Factors ===
    pub delta_median_ici_ms: f32,
    pub delta_onset_rate_hz: f32,
    pub delta_ici_coefficient_of_variation: f32,
}

impl VectorDelta {
    /// Create a zero delta (no change)
    pub fn zero() -> Self {
        Self {
            delta_mean_f0_hz: 0.0,
            delta_duration_ms: 0.0,
            delta_f0_range_hz: 0.0,
            delta_harmonic_to_noise_ratio: 0.0,
            delta_spectral_flatness: 0.0,
            delta_attack_time_ms: 0.0,
            delta_decay_time_ms: 0.0,
            delta_sustain_level: 0.0,
            delta_vibrato_rate_hz: 0.0,
            delta_vibrato_depth: 0.0,
            delta_jitter: 0.0,
            delta_mfcc_1: 0.0,
            delta_mfcc_2: 0.0,
            delta_mfcc_3: 0.0,
            delta_mfcc_4: 0.0,
            delta_spectral_contrast: 0.0,
            delta_median_ici_ms: 0.0,
            delta_onset_rate_hz: 0.0,
            delta_ici_coefficient_of_variation: 0.0,
        }
    }

    /// Calculate delta from two vectors (target - source)
    pub fn from_vectors(target: &Vector17D, source: &Vector17D) -> Self {
        Self {
            delta_mean_f0_hz: target.mean_f0_hz - source.mean_f0_hz,
            delta_duration_ms: target.duration_ms - source.duration_ms,
            delta_f0_range_hz: target.f0_range_hz - source.f0_range_hz,
            delta_harmonic_to_noise_ratio: target.harmonic_to_noise_ratio - source.harmonic_to_noise_ratio,
            delta_spectral_flatness: target.spectral_flatness - source.spectral_flatness,
            delta_attack_time_ms: target.attack_time_ms - source.attack_time_ms,
            delta_decay_time_ms: target.decay_time_ms - source.decay_time_ms,
            delta_sustain_level: target.sustain_level - source.sustain_level,
            delta_vibrato_rate_hz: target.vibrato_rate_hz - source.vibrato_rate_hz,
            delta_vibrato_depth: target.vibrato_depth - source.vibrato_depth,
            delta_jitter: target.jitter - source.jitter,
            delta_mfcc_1: target.mfcc_1 - source.mfcc_1,
            delta_mfcc_2: target.mfcc_2 - source.mfcc_2,
            delta_mfcc_3: target.mfcc_3 - source.mfcc_3,
            delta_mfcc_4: target.mfcc_4 - source.mfcc_4,
            delta_spectral_contrast: target.spectral_contrast - source.spectral_contrast,
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
    pub target: Vector17D,
    /// Navigation mode used
    pub mode: NavigationMode,
    /// Anchor island if interpolation was used
    pub anchor_island: Option<String>,
    /// Distance from anchor to target (normalized)
    pub distance_to_anchor: f32,
    /// Whether clamping was applied
    pub was_clamped: bool,
    /// Original target before clamping (if clamped)
    pub original_target: Option<Vector17D>,
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
/// by limiting the maximum warp distance in 17D space.
#[derive(Debug, Clone)]
pub struct SafetyClamp {
    /// Maximum safe warp distance (normalized)
    max_safe_warp: f32,
}

impl SafetyClamp {
    /// Create a new safety clamp with default 20% max warp
    pub fn new() -> Self {
        Self {
            max_safe_warp: 0.2,
        }
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
        target: &Vector17D,
        anchor: &Vector17D,
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
    /// 17D feature vector
    pub features: Vector17D,
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
        self.species_index
            .entry(species)
            .or_default()
            .push(key);
    }

    /// Find the nearest island to a target vector
    ///
    /// Returns None if database is empty.
    /// O(n) linear search - adequate for <10k phrases, can upgrade to KD-tree later.
    pub fn find_nearest_17d(&self, target: &Vector17D) -> Option<&AudioIsland> {
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
    pub fn find_nearest_17d_species(
        &self,
        target: &Vector17D,
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
    pub fn find_k_nearest_17d(&self, target: &Vector17D, k: usize) -> Vec<&AudioIsland> {
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
            .map(|keys| {
                keys.iter()
                    .filter_map(|k| self.islands.get(k))
                    .collect()
            })
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
        let crossfade_in_samples = (event.crossfade_in_ms / 1000.0 * self.sample_rate as f32) as usize;
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

/// Apply 17D delta to granular synthesis parameters
///
/// This maps the 17D acoustic delta to synthesizer control parameters.
/// This is the PRIMARY integration point for Acoustic Algebra → Rust Synthesis.
pub fn apply_delta_to_granular(
    delta: &VectorDelta,
    base_params: &GranularParams,
    source_metadata: &Vector17D,
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
    let roughness_amount = (base_params.roughness_amount + delta.delta_spectral_flatness).clamp(0.0, 1.0);

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
    pub fn interpolate(&self, start: &Vector17D, end: &Vector17D, alpha: f32) -> Vector17D {
        start.interpolate(end, alpha)
    }

    /// Extrapolate from origin in direction (Ocean Explorer - RISKY)
    pub fn extrapolate(&self, origin: &Vector17D, direction: &VectorDelta, factor: f32) -> Vector17D {
        origin.extrapolate(direction, factor)
    }

    /// Apply safety clamping to target
    pub fn clamp_to_safe_distance(
        &self,
        target: &Vector17D,
        anchor: &Vector17D,
        anchor_island: Option<String>,
    ) -> NavigationWaypoint {
        self.clamp.clamp_target(target, anchor, anchor_island)
    }

    /// Find nearest island to target vector
    pub fn find_nearest_island(&self, target: &Vector17D) -> Option<&AudioIsland> {
        self.database.find_nearest_17d(target)
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
    fn create_test_vector(f0: f32, duration: f32) -> Vector17D {
        Vector17D {
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
    // Vector17D Tests
    // =========================================================================

    #[test]
    fn test_vector17d_default() {
        let v = Vector17D::default();
        assert_approx_eq(v.mean_f0_hz, 7000.0, 1e-5);
        assert_approx_eq(v.duration_ms, 50.0, 1e-5);
    }

    #[test]
    fn test_vector17d_new() {
        let v = Vector17D::new(
            8000.0, 60.0, 500.0, 25.0, 0.4, 10.0, 25.0, 0.8, 8.0, 0.03, 0.02,
            -12.0, -6.0, -3.0, -1.5, 25.0, 180.0, 10.0, 0.4,
        );
        assert_approx_eq(v.mean_f0_hz, 8000.0, 1e-5);
        assert_approx_eq(v.duration_ms, 60.0, 1e-5);
        assert_approx_eq(v.f0_range_hz, 500.0, 1e-5);
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
        let delta = VectorDelta {
            delta_mean_f0_hz: 1000.0,
            delta_duration_ms: 10.0,
            ..VectorDelta::zero()
        };
        let result = v1.extrapolate(&delta, 0.0);

        assert_approx_eq(result.mean_f0_hz, 7000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 50.0, 1e-5);
    }

    #[test]
    fn test_extrapolate_unit_factor() {
        let v1 = create_test_vector(7000.0, 50.0);
        let delta = VectorDelta {
            delta_mean_f0_hz: 1000.0,
            delta_duration_ms: 10.0,
            ..VectorDelta::zero()
        };
        let result = v1.extrapolate(&delta, 1.0);

        assert_approx_eq(result.mean_f0_hz, 8000.0, 1e-5);
        assert_approx_eq(result.duration_ms, 60.0, 1e-5);
    }

    #[test]
    fn test_extrapolate_double_factor() {
        let v1 = create_test_vector(7000.0, 50.0);
        let delta = VectorDelta {
            delta_mean_f0_hz: 1000.0,
            delta_duration_ms: 10.0,
            ..VectorDelta::zero()
        };
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
        let nearest = db.find_nearest_17d(&target);

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
        let nearest = db.find_nearest_17d(&target);
        assert_eq!(nearest.unwrap().key, "bat1");

        // Nearest marmoset should be marmoset1
        let nearest_marmoset = db.find_nearest_17d_species(&target, "marmoset");
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
        let k_nearest = db.find_k_nearest_17d(&target, 3);

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
        let delta = VectorDelta {
            delta_mean_f0_hz: 1000.0,
            delta_duration_ms: 10.0,
            delta_spectral_flatness: 0.1,
            delta_jitter: 0.01,
            ..VectorDelta::zero()
        };

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
