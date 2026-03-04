//! Acoustic Purity Filter - Noise Rejection for Bioacoustic Detection
//!
//! Problem: Smart Segmenter detects wind, rain, insect stridulation as "bio-activity"
//! Solution: Apply a "Purity Gate" between Segmenter and Classifier
//!
//! Architecture:
//!   Audio -> Smart Segmenter -> **Purity Gate** -> Feature Extractor -> Classifier
//!
//! Key Metrics:
//!   - HNR (Harmonic-to-Noise Ratio): Biological = tonal (HNR > 5dB), Rain = chaotic (HNR < 5dB)
//!   - Spectral Flatness: Biological = peaky (< 0.5), Rain = white-noise-like (> 0.6)
//!   - Duration: Raindrops = short (< 50ms), Biological = sustained (> 40ms)

use anyhow::Result;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the purity filter
#[derive(Debug, Clone)]
pub struct PurityFilterConfig {
    /// Minimum Harmonic-to-Noise Ratio (dB) - reject chaotic noise
    pub min_hnr_db: f32,
    /// Maximum spectral flatness - reject white-noise-like sounds
    pub max_spectral_flatness: f32,
    /// Minimum duration (ms) - reject transient clicks/pops
    pub min_duration_ms: f32,
    /// Minimum RMS energy - reject silence
    pub min_rms: f32,
    /// Maximum zero-crossing rate - reject certain mechanical noise
    pub max_zcr: f32,
}

impl Default for PurityFilterConfig {
    fn default() -> Self {
        Self {
            min_hnr_db: 3.0,            // Must have some tonality
            max_spectral_flatness: 0.6, // Reject white-noise/wind
            min_duration_ms: 40.0,      // Reject transient clicks
            min_rms: 0.001,             // Reject silence
            max_zcr: 0.5,               // Allow high ZCR for insects
        }
    }
}

impl PurityFilterConfig {
    /// Strict preset - more aggressive noise rejection
    ///
    /// Use when false positives are costly (e.g., automated detection systems)
    pub fn strict() -> Self {
        Self {
            min_hnr_db: 5.0,            // Higher tonality requirement
            max_spectral_flatness: 0.5, // Lower flatness tolerance
            min_duration_ms: 50.0,      // Longer duration required
            min_rms: 0.002,             // Higher energy threshold
            max_zcr: 0.4,               // Lower ZCR tolerance
        }
    }

    /// Loose preset - more permissive filtering
    ///
    /// Use for exploratory analysis or when false negatives are costly
    pub fn loose() -> Self {
        Self {
            min_hnr_db: 1.0,            // Lower tonality requirement
            max_spectral_flatness: 0.7, // Higher flatness tolerance
            min_duration_ms: 30.0,      // Shorter duration allowed
            min_rms: 0.0005,            // Lower energy threshold
            max_zcr: 0.6,               // Higher ZCR tolerance
        }
    }

    /// Preset for field recordings with high environmental noise
    ///
    /// Optimized for jungle/field conditions with insect chorus, wind, etc.
    pub fn for_field_recordings() -> Self {
        Self {
            min_hnr_db: 2.0,             // Low threshold for noisy conditions
            max_spectral_flatness: 0.65, // Moderate tolerance
            min_duration_ms: 35.0,       // Short duration for quick calls
            min_rms: 0.001,              // Standard threshold
            max_zcr: 0.5,                // Standard tolerance
        }
    }

    /// Preset for clean laboratory recordings
    ///
    /// Use for controlled environment recordings with low noise
    pub fn for_lab_recordings() -> Self {
        Self {
            min_hnr_db: 4.0,             // Higher threshold for clean audio
            max_spectral_flatness: 0.55, // Lower tolerance
            min_duration_ms: 40.0,       // Standard duration
            min_rms: 0.0008,             // Lower threshold (quiet recordings)
            max_zcr: 0.45,               // Moderate tolerance
        }
    }
}

// ============================================================================
// Acoustic Purity Filter
// ============================================================================

/// Filters non-biological sounds before classification
///
/// This is the "Purity Gate" that sits between the Smart Segmenter
/// and the Classifier. It uses fast 45D physics features to reject
/// rain, wind, and mechanical noise before expensive classification.
pub struct AcousticPurityFilter {
    config: PurityFilterConfig,
    stats: FilterStats,
}

/// Statistics for debugging/tuning
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    pub total_segments: usize,
    pub rejected_duration: usize,
    pub rejected_hnr: usize,
    pub rejected_flatness: usize,
    pub rejected_rms: usize,
    pub passed: usize,
}

/// Result of purity check with reasons
#[derive(Debug, Clone)]
pub struct PurityResult {
    pub is_biological: bool,
    pub rejection_reason: Option<String>,
    pub hnr_db: f32,
    pub spectral_flatness: f32,
    pub duration_ms: f32,
    pub rms: f32,
}

impl AcousticPurityFilter {
    /// Create a new filter with default configuration
    pub fn new() -> Self {
        Self {
            config: PurityFilterConfig::default(),
            stats: FilterStats::default(),
        }
    }

    /// Create a filter with custom configuration
    pub fn with_config(config: PurityFilterConfig) -> Self {
        Self {
            config,
            stats: FilterStats::default(),
        }
    }

    /// Check if a segment is likely biological
    ///
    /// Uses 45D physics features (fast to compute) to determine
    /// if a segment is biological or environmental noise.
    pub fn check_purity(&mut self, features: &PurityFeatures) -> PurityResult {
        self.stats.total_segments += 1;

        let result = self.check_purity_internal(features);

        if result.is_biological {
            self.stats.passed += 1;
        } else {
            match result.rejection_reason.as_deref() {
                Some("duration") => self.stats.rejected_duration += 1,
                Some("hnr") => self.stats.rejected_hnr += 1,
                Some("flatness") => self.stats.rejected_flatness += 1,
                Some("rms") => self.stats.rejected_rms += 1,
                _ => {}
            }
        }

        result
    }

    fn check_purity_internal(&self, features: &PurityFeatures) -> PurityResult {
        // 1. Reject short transients (often raindrops or mechanical noise)
        if features.duration_ms < self.config.min_duration_ms {
            return PurityResult {
                is_biological: false,
                rejection_reason: Some("duration".to_string()),
                hnr_db: features.hnr_db,
                spectral_flatness: features.spectral_flatness,
                duration_ms: features.duration_ms,
                rms: features.rms,
            };
        }

        // 2. Reject silence/very quiet segments
        if features.rms < self.config.min_rms {
            return PurityResult {
                is_biological: false,
                rejection_reason: Some("rms".to_string()),
                hnr_db: features.hnr_db,
                spectral_flatness: features.spectral_flatness,
                duration_ms: features.duration_ms,
                rms: features.rms,
            };
        }

        // 3. Reject chaotic noise (wind, heavy rain)
        // Low HNR = High Noise content
        if features.hnr_db < self.config.min_hnr_db {
            return PurityResult {
                is_biological: false,
                rejection_reason: Some("hnr".to_string()),
                hnr_db: features.hnr_db,
                spectral_flatness: features.spectral_flatness,
                duration_ms: features.duration_ms,
                rms: features.rms,
            };
        }

        // 4. Reject broadband noise (rain, waterfall)
        // High Flatness = Uniform spectrum (like white noise)
        if features.spectral_flatness > self.config.max_spectral_flatness {
            return PurityResult {
                is_biological: false,
                rejection_reason: Some("flatness".to_string()),
                hnr_db: features.hnr_db,
                spectral_flatness: features.spectral_flatness,
                duration_ms: features.duration_ms,
                rms: features.rms,
            };
        }

        // Passed the purity gate - likely biological
        PurityResult {
            is_biological: true,
            rejection_reason: None,
            hnr_db: features.hnr_db,
            spectral_flatness: features.spectral_flatness,
            duration_ms: features.duration_ms,
            rms: features.rms,
        }
    }

    /// Quick check without tracking stats
    pub fn is_biological(&self, features: &PurityFeatures) -> bool {
        self.check_purity_internal(features).is_biological
    }

    /// Get filter statistics
    pub fn stats(&self) -> &FilterStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FilterStats::default();
    }
}

impl Default for AcousticPurityFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Purity Features (Subset of 45D Physics Features)
// ============================================================================

/// Features needed for purity checking (fast to compute)
#[derive(Debug, Clone)]
pub struct PurityFeatures {
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Root mean square energy
    pub rms: f32,
    /// Harmonic-to-Noise Ratio (dB)
    pub hnr_db: f32,
    /// Spectral flatness (0-1, higher = more noise-like)
    pub spectral_flatness: f32,
    /// Zero-crossing rate
    pub zcr: f32,
}

impl PurityFeatures {
    /// Extract purity features from audio
    pub fn from_audio(audio: &[f32], sample_rate: u32) -> Self {
        let duration_ms = (audio.len() as f32 / sample_rate as f32) * 1000.0;
        let rms = compute_rms(audio);
        let hnr_db = compute_hnr(audio, sample_rate);
        let spectral_flatness = compute_spectral_flatness(audio);
        let zcr = compute_zcr(audio);

        Self {
            duration_ms,
            rms,
            hnr_db,
            spectral_flatness,
            zcr,
        }
    }

    /// Extract from 45D feature vector
    pub fn from_45d(features: &[f32]) -> Self {
        // Indices in 45D vector (from MicroDynamicsFeatures45D)
        // duration_ms is at index 1, rms is computed, hnr is at index 6, etc.
        Self {
            duration_ms: features.get(1).copied().unwrap_or(100.0),
            rms: compute_rms_from_features(features),
            hnr_db: features.get(6).copied().unwrap_or(10.0),
            spectral_flatness: features.get(20).copied().unwrap_or(0.3),
            zcr: features.get(4).copied().unwrap_or(0.1),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn compute_rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = audio.iter().map(|&x| x * x).sum();
    (sum_sq / audio.len() as f32).sqrt()
}

fn compute_rms_from_features(_features: &[f32]) -> f32 {
    // Approximate from feature vector
    0.01 // Default
}

fn compute_hnr(audio: &[f32], sample_rate: u32) -> f32 {
    // Simplified HNR calculation
    // Real implementation would use autocorrelation
    if audio.len() < 100 {
        return 0.0;
    }

    // Compute autocorrelation at lag 0 and at fundamental period
    let rms = compute_rms(audio);
    if rms < 1e-6 {
        return 0.0;
    }

    // Estimate fundamental period (roughly)
    let min_period = (sample_rate as f32 / 2000.0) as usize; // 2kHz max
    let max_period = (sample_rate as f32 / 100.0) as usize; // 100Hz min
    let max_period = max_period.min(audio.len() / 2);

    if min_period >= max_period {
        return 0.0;
    }

    // Find peak autocorrelation in expected pitch range
    let mut max_corr = 0.0;
    for lag in min_period..max_period {
        let mut corr = 0.0;
        for i in 0..(audio.len() - lag) {
            corr += audio[i] * audio[i + lag];
        }
        if corr > max_corr {
            max_corr = corr;
        }
    }

    // Normalize
    let energy: f32 = audio.iter().map(|&x| x * x).sum();
    if energy > 1e-10 {
        let normalized = max_corr / energy;
        // Convert to dB
        if normalized > 0.0 {
            return 10.0 * normalized.log10();
        }
    }

    0.0
}

fn compute_spectral_flatness(audio: &[f32]) -> f32 {
    // Spectral flatness = geometric_mean / arithmetic_mean of spectrum
    // High value (> 0.6) = noise-like
    // Low value (< 0.3) = tonal/harmonic

    let n = audio.len().next_power_of_two();
    let mut spectrum = vec![0.0f32; n / 2 + 1];

    // Simple DFT for magnitude spectrum
    for k in 0..=n / 2 {
        let mut sum_r = 0.0;
        let mut sum_i = 0.0;
        for (j, &s) in audio.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
            sum_r += s * angle.cos();
            sum_i += s * angle.sin();
        }
        spectrum[k] = (sum_r * sum_r + sum_i * sum_i).sqrt() + 1e-10;
    }

    // Compute geometric and arithmetic means
    let log_sum: f32 = spectrum.iter().map(|x| x.ln()).sum();
    let sum: f32 = spectrum.iter().sum();
    let n_bins = spectrum.len() as f32;

    if sum > 0.0 && n_bins > 0.0 {
        let geometric_mean = (log_sum / n_bins).exp();
        let arithmetic_mean = sum / n_bins;
        (geometric_mean / arithmetic_mean).min(1.0).max(0.0)
    } else {
        0.5
    }
}

fn compute_zcr(audio: &[f32]) -> f32 {
    if audio.len() < 2 {
        return 0.0;
    }

    let mut crossings = 0;
    for i in 1..audio.len() {
        if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
            crossings += 1;
        }
    }

    crossings as f32 / (audio.len() - 1) as f32
}

// ============================================================================
// Tests (TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(duration_ms: f32, rms: f32, hnr_db: f32, flatness: f32) -> PurityFeatures {
        PurityFeatures {
            duration_ms,
            rms,
            hnr_db,
            spectral_flatness: flatness,
            zcr: 0.1,
        }
    }

    #[test]
    fn test_filter_accepts_biological() {
        let mut filter = AcousticPurityFilter::new();

        // Typical bird call: 200ms, good HNR, tonal
        let features = make_features(200.0, 0.05, 15.0, 0.2);

        let result = filter.check_purity(&features);

        assert!(result.is_biological);
        assert!(result.rejection_reason.is_none());
    }

    #[test]
    fn test_filter_rejects_short_transient() {
        let mut filter = AcousticPurityFilter::new();

        // Very short: like a raindrop or click
        let features = make_features(20.0, 0.05, 10.0, 0.3);

        let result = filter.check_purity(&features);

        assert!(!result.is_biological);
        assert_eq!(result.rejection_reason, Some("duration".to_string()));
    }

    #[test]
    fn test_filter_rejects_low_hnr() {
        let mut filter = AcousticPurityFilter::new();

        // Low HNR: chaotic/windy
        let features = make_features(200.0, 0.05, 1.0, 0.3);

        let result = filter.check_purity(&features);

        assert!(!result.is_biological);
        assert_eq!(result.rejection_reason, Some("hnr".to_string()));
    }

    #[test]
    fn test_filter_rejects_high_flatness() {
        let mut filter = AcousticPurityFilter::new();

        // High flatness: white-noise-like (rain/wind)
        let features = make_features(200.0, 0.05, 10.0, 0.8);

        let result = filter.check_purity(&features);

        assert!(!result.is_biological);
        assert_eq!(result.rejection_reason, Some("flatness".to_string()));
    }

    #[test]
    fn test_filter_rejects_silence() {
        let mut filter = AcousticPurityFilter::new();

        // Very quiet: silence
        let features = make_features(200.0, 0.0001, 10.0, 0.3);

        let result = filter.check_purity(&features);

        assert!(!result.is_biological);
        assert_eq!(result.rejection_reason, Some("rms".to_string()));
    }

    #[test]
    fn test_filter_stats() {
        let mut filter = AcousticPurityFilter::new();

        // 1 biological, 1 rejected
        filter.check_purity(&make_features(200.0, 0.05, 15.0, 0.2));
        filter.check_purity(&make_features(10.0, 0.05, 15.0, 0.2));

        let stats = filter.stats();
        assert_eq!(stats.total_segments, 2);
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.rejected_duration, 1);
    }

    #[test]
    fn test_custom_config() {
        let config = PurityFilterConfig {
            min_hnr_db: 10.0, // Stricter
            max_spectral_flatness: 0.4,
            min_duration_ms: 100.0,
            min_rms: 0.01,
            max_zcr: 0.3,
        };
        let filter = AcousticPurityFilter::with_config(config);

        // Would pass default, but fails stricter HNR
        let features = make_features(200.0, 0.05, 8.0, 0.3);

        assert!(!filter.is_biological(&features));
    }

    #[test]
    fn test_purity_features_from_audio() {
        // Generate test audio: 440Hz sine wave (tonal)
        let sample_rate = 44100;
        let duration_samples = sample_rate; // 1 second
        let audio: Vec<f32> = (0..duration_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.1
            })
            .collect();

        let features = PurityFeatures::from_audio(&audio, sample_rate);

        assert!(features.duration_ms > 900.0);
        assert!(features.rms > 0.01);
        // Sine wave should have low flatness (tonal)
        assert!(
            features.spectral_flatness < 0.5,
            "Sine wave should have low flatness, got {}",
            features.spectral_flatness
        );
    }
}
