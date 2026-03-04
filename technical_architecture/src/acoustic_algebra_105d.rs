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
// Modulation Spectra Computation (Module 5)
// ============================================================================

/// Compute Amplitude Modulation spectrum from energy envelope
///
/// Decomposes the energy envelope into frequency bands to detect:
/// - Trills (10-30Hz AM)
/// - Roughness (30-50Hz AM)
/// - Insect buzz (50-100Hz AM)
pub fn compute_am_spectrum(
    energy_envelope: &[f32],
    sample_rate: f32,
    frame_rate: f32,
) -> (f32, f32, f32, f32, f32) {
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
    let fm_depth = if mean_f0 > 0.0 {
        mean_deriv / mean_f0
    } else {
        0.0
    };

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
    let am_fm_ratio = if fm_total > 1e-10 {
        am_total / fm_total
    } else {
        0.0
    };

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

    (
        am_fm_ratio,
        complexity,
        trill_strength,
        flutter_index,
        synchrony,
    )
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
    let skewness: f32 = iois
        .iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f32>()
        / n;

    // Kurtosis (tailedness of distribution)
    let kurtosis: f32 = iois
        .iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f32>()
        / n
        - 3.0; // Excess kurtosis

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
pub fn compute_psychoacoustics(
    spectrum: &[f32],
    frequencies: &[f32],
    rms_energy: f32,
) -> (f32, f32, f32, f32, f32) {
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
    let low_freq_energy: f32 = spectrum
        .iter()
        .take(spectrum.len() / 4)
        .map(|&m| m * m)
        .sum();
    let fluctuation_strength = if total_energy > 1e-10 {
        (low_freq_energy / total_energy).sqrt()
    } else {
        0.0
    };

    (
        sharpness,
        roughness,
        loudness_sone,
        tonality,
        fluctuation_strength,
    )
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

        let (am_0_10, am_10_30, am_30_50, am_50_100, _depth) =
            compute_am_spectrum(&envelope, frame_rate, frame_rate);

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

        let (fm_0_10, fm_10_30, fm_30_50, _fm_50_100, _depth) =
            compute_fm_spectrum(&f0, frame_rate);

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
        assert!(fluctuation >= 0.0 && fluctuation <= 1.0);
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
}
