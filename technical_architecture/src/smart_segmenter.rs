//! Smart Segmentation Router: Precision Ethology
//! ==============================================
//!
//! Implements intelligent routing between CPD (Change Point Detection) and NBD
//! (Neural Boundary Detection) based on the "Linguistic Profile" of the audio.
//!
//! ## Key Discovery: Duration CV as Linguistic Complexity Proxy
//!
//! The Duration Coefficient of Variation (CV) reveals the underlying communication type:
//! - **Low CV (~0.26-0.30)**: Uniform segments = Crystallized Song (songbirds, insects)
//! - **High CV (~0.40-0.95)**: Variable segments = Graded Calls (primates, bats)
//!
//! ## Architecture
//! ```text
//! INPUT: Audio Buffer
//!         ↓
//! [Fast Preliminary Scan (500ms)]
//!         ↓
//! ┌───────────────────────────────────────┐
//! │ Linguistic Profile Detection           │
//! │ - Energy Variance                      │
//! │ - Rhythmicity Index                    │
//! │ - Spectral Stability                   │
//! └───────────────────────────────────────┘
//!         ↓
//! ┌─────────────┬─────────────┐
//! ↓             ↓             ↓
//! CPD         NBD         HYBRID
//! (Rhythmic)  (Graded)    (Mixed)
//! ↓             ↓             ↓
//! [Zipf Quality Check]
//! ↓             ↓             ↓
//! PhraseBoundaries
//! ```
//!
//! ## Usage
//! ```rust
//! use technical_architecture::SmartSegmenter;
//!
//! let segmenter = SmartSegmenter::new(44100);
//! let boundaries = segmenter.segment_smart(&audio);
//! println!("Method used: {:?}", boundaries.method);
//! ```

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::dynamic_segmenter::{DynamicSegmenter, DynamicSegmenterConfig};
use crate::neural_boundary::{
    BoundaryDetectorConfig, BoundaryType, NeuralBoundaryDetector, PhraseBoundary,
};

// ============================================================================
// Configuration
// ============================================================================

/// Forced segmentation mode for explicit control
///
/// # Migration Guide (v2.1.0)
///
/// NBD (Neural Boundary Detection) is now the **default** segmentation method
/// because it correctly handles:
/// - Graded signals (primates, bats, cetaceans)
/// - Continuous streams without artificial boundaries
/// - Species where the syllable (not call) is the atomic unit
///
/// EBD/CPD should only be used for **crystallized songs** (songbirds like Zebra Finches).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ForcedSegmentationMode {
    /// Automatically select based on linguistic profile (default)
    /// Defaults to NBD unless high-confidence crystallized song detected
    #[default]
    Auto,
    /// Force NBD - recommended for graded signals (mammals)
    ForceNBD,
    /// Force CPD/EBD - only for crystallized songs (songbirds)
    /// WARNING: Will fragment graded signals incorrectly
    ForceCPD,
    /// Force hybrid - combine both methods
    ForceHybrid,
}

/// Configuration for the Smart Segmenter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartSegmenterConfig {
    /// Sample rate
    pub sample_rate: u32,
    /// Duration of preliminary scan in milliseconds
    pub scan_duration_ms: f32,
    /// Energy variance threshold for "rhythmic" classification
    pub energy_variance_threshold: f32,
    /// Rhythmicity threshold for CPD routing
    pub rhythmicity_threshold: f32,
    /// Duration CV threshold for NBD routing
    pub duration_cv_threshold: f32,
    /// Enable self-tuning with Zipf correlation check
    pub enable_self_tuning: bool,
    /// Minimum Zipf R² for quality validation
    pub min_zipf_r2: f32,
    /// Forced segmentation mode (default: Auto -> NBD)
    pub forced_mode: ForcedSegmentationMode,
    /// Minimum confidence for crystallized song to use CPD (default: 0.85)
    /// Only high-confidence crystallized songs will route to CPD in Auto mode
    pub crystallized_confidence_threshold: f32,
}

impl Default for SmartSegmenterConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            scan_duration_ms: 500.0,
            energy_variance_threshold: 0.15,
            rhythmicity_threshold: 0.6,
            duration_cv_threshold: 0.35,
            enable_self_tuning: true,
            min_zipf_r2: 0.85,
            forced_mode: ForcedSegmentationMode::Auto,
            crystallized_confidence_threshold: 0.85, // High bar for CPD usage
        }
    }
}

// ============================================================================
// Linguistic Profile
// ============================================================================

/// Detected linguistic profile of the audio
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinguisticProfile {
    /// Crystallized song: rhythmic, stereotyped (songbirds, insects)
    CrystallizedSong,
    /// Graded calls: variable duration, conversational (primates, bats)
    GradedCall,
    /// Cultural coda: rhythmic but learned (sperm whale clicks)
    CulturalCoda,
    /// Mixed: contains both rhythmic and graded elements
    Mixed,
}

/// Analysis results from preliminary scan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreliminaryAnalysis {
    /// Energy variance across frames
    pub energy_variance: f32,
    /// Rhythmicity index (0.0-1.0)
    pub rhythmicity: f32,
    /// Spectral stability (0.0-1.0)
    pub spectral_stability: f32,
    /// Duration coefficient of variation estimate
    pub estimated_duration_cv: f32,
    /// Detected linguistic profile
    pub profile: LinguisticProfile,
    /// Confidence in the profile detection (0.0-1.0)
    pub confidence: f32,
}

/// Segmentation method used
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentationMethod {
    /// Change Point Detection (CPD) - for rhythmic signals
    CPD,
    /// Neural Boundary Detection (NBD) - for graded signals
    NBD,
    /// Hybrid approach - both methods combined
    Hybrid,
}

/// Result from smart segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartSegmentationResult {
    /// Detected phrase boundaries
    pub boundaries: Vec<PhraseBoundary>,
    /// Method used for segmentation
    pub method: SegmentationMethod,
    /// Preliminary analysis that led to the routing decision
    pub analysis: PreliminaryAnalysis,
    /// Zipf correlation quality metric (if self-tuning enabled)
    pub zipf_r2: Option<f32>,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

// ============================================================================
// Smart Segmenter
// ============================================================================

/// Intelligent segmentation router that selects optimal method based on
/// the linguistic characteristics of the audio.
pub struct SmartSegmenter {
    config: SmartSegmenterConfig,
    cpd_segmenter: DynamicSegmenter,
    nbd_detector: NeuralBoundaryDetector,
}

impl SmartSegmenter {
    /// Create a new smart segmenter with default configuration
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(SmartSegmenterConfig {
            sample_rate,
            ..Default::default()
        })
    }

    /// Create a smart segmenter with custom configuration
    pub fn with_config(config: SmartSegmenterConfig) -> Self {
        let sample_rate = config.sample_rate;

        let cpd_config = DynamicSegmenterConfig {
            frame_duration_ms: 10.0,
            min_phrase_duration_ms: 30.0,
            max_phrase_duration_ms: 2000.0,
            change_threshold: 0.25,
            smoothing_window: 3,
            peak_prominence: 0.05,
            feature_dim: 45,
        };

        let nbd_config = BoundaryDetectorConfig {
            hop_size: 512,
            sample_rate,
            min_phrase_duration_ms: 50.0,
            threshold: 0.5,
            smoothing_frames: 3,
        };

        Self {
            config,
            cpd_segmenter: DynamicSegmenter::new(cpd_config, sample_rate),
            nbd_detector: NeuralBoundaryDetector::with_config(nbd_config),
        }
    }

    /// Perform smart segmentation with automatic method selection
    pub fn segment_smart(&mut self, audio: &[f32]) -> SmartSegmentationResult {
        let start = Instant::now();

        // Step 1: Fast preliminary scan
        let scan_samples =
            ((self.config.scan_duration_ms / 1000.0) * self.config.sample_rate as f32) as usize;
        let scan_audio = if audio.len() > scan_samples {
            &audio[..scan_samples]
        } else {
            audio
        };

        let analysis = self.analyze_preliminary(scan_audio);

        // Step 2: Route to appropriate method
        let (boundaries, method) = self.route_and_segment(audio, &analysis);

        // Step 3: Self-tuning check (if enabled)
        let zipf_r2 = if self.config.enable_self_tuning {
            let r2 = self.compute_zipf_correlation(&boundaries);
            if r2 < self.config.min_zipf_r2 {
                // Quality check failed, try alternate method
                let alternate_method = match method {
                    SegmentationMethod::CPD => SegmentationMethod::NBD,
                    SegmentationMethod::NBD => SegmentationMethod::CPD,
                    SegmentationMethod::Hybrid => SegmentationMethod::Hybrid,
                };

                let (alt_boundaries, _) = self.segment_with_method(audio, alternate_method);
                let alt_r2 = self.compute_zipf_correlation(&alt_boundaries);

                if alt_r2 > r2 {
                    return SmartSegmentationResult {
                        boundaries: alt_boundaries,
                        method: alternate_method,
                        analysis,
                        zipf_r2: Some(alt_r2),
                        processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
                    };
                }
            }
            Some(r2)
        } else {
            None
        };

        SmartSegmentationResult {
            boundaries,
            method,
            analysis,
            zipf_r2,
            processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        }
    }

    /// Analyze the initial portion of audio to determine linguistic profile
    fn analyze_preliminary(&self, audio: &[f32]) -> PreliminaryAnalysis {
        let hop_size = 512;
        let n_frames = audio.len() / hop_size;

        if n_frames < 2 {
            return PreliminaryAnalysis {
                energy_variance: 0.0,
                rhythmicity: 0.0,
                spectral_stability: 0.0,
                estimated_duration_cv: 0.0,
                profile: LinguisticProfile::Mixed,
                confidence: 0.0,
            };
        }

        // Compute energy profile
        let energy_profile: Vec<f32> = (0..n_frames)
            .map(|i| {
                let start = i * hop_size;
                let end = (start + hop_size).min(audio.len());
                let frame = &audio[start..end];
                let sum: f32 = frame.iter().map(|x| x * x).sum();
                (sum / frame.len() as f32).sqrt()
            })
            .collect();

        // Compute energy variance
        let mean_energy: f32 = energy_profile.iter().sum::<f32>() / energy_profile.len() as f32;
        let energy_variance: f32 = energy_profile
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energy_profile.len() as f32;

        // Compute rhythmicity (autocorrelation at typical beat intervals)
        let rhythmicity = self.compute_rhythmicity(&energy_profile);

        // Compute spectral stability
        let spectral_stability = self.compute_spectral_stability(audio, hop_size);

        // Estimate duration CV from energy peaks
        let estimated_duration_cv = self.estimate_duration_cv(&energy_profile);

        // Determine profile
        let (profile, confidence) = self.classify_profile(
            energy_variance,
            rhythmicity,
            spectral_stability,
            estimated_duration_cv,
        );

        PreliminaryAnalysis {
            energy_variance,
            rhythmicity,
            spectral_stability,
            estimated_duration_cv,
            profile,
            confidence,
        }
    }

    /// Compute rhythmicity using autocorrelation
    fn compute_rhythmicity(&self, energy: &[f32]) -> f32 {
        if energy.len() < 10 {
            return 0.0;
        }

        // Look for periodicity at typical beat intervals (50-500ms)
        let min_lag = 5; // ~50ms at 100Hz frame rate
        let max_lag = 50; // ~500ms

        let mean: f32 = energy.iter().sum::<f32>() / energy.len() as f32;
        let variance: f32 = energy.iter().map(|e| (e - mean).powi(2)).sum::<f32>();

        if variance < 1e-10 {
            return 0.0;
        }

        let mut max_autocorr: f32 = 0.0;
        for lag in min_lag..=max_lag.min(energy.len() / 2) {
            let autocorr: f32 = energy[..energy.len() - lag]
                .iter()
                .zip(energy[lag..].iter())
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f32>()
                / variance;

            max_autocorr = max_autocorr.max(autocorr);
        }

        max_autocorr.max(0.0).min(1.0)
    }

    /// Compute spectral stability (how much spectral content changes over time)
    fn compute_spectral_stability(&self, audio: &[f32], hop_size: usize) -> f32 {
        let fft_size = 1024;
        let n_frames = audio.len() / hop_size;

        if n_frames < 2 {
            return 1.0;
        }

        // Compute spectral centroids for each frame
        let centroids: Vec<f32> = (0..n_frames.min(20))
            .map(|i| {
                let start = i * hop_size;
                let end = (start + fft_size).min(audio.len());
                let frame = &audio[start..end];

                // Simple spectral centroid approximation using zero crossings
                let mut crossings = 0;
                for j in 1..frame.len() {
                    if frame[j] * frame[j - 1] < 0.0 {
                        crossings += 1;
                    }
                }
                crossings as f32 / frame.len() as f32
            })
            .collect();

        // Stability = low variance in spectral centroid
        let mean: f32 = centroids.iter().sum::<f32>() / centroids.len() as f32;
        let variance: f32 =
            centroids.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / centroids.len() as f32;

        1.0 / (1.0 + variance * 100.0)
    }

    /// Estimate duration CV from energy peaks
    fn estimate_duration_cv(&self, energy: &[f32]) -> f32 {
        // Find peaks in energy profile
        let mut peak_intervals: Vec<f32> = Vec::new();
        let mut last_peak_idx: Option<usize> = None;

        let threshold: f32 = energy.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 0.5;

        for i in 1..energy.len() - 1 {
            if energy[i] > threshold && energy[i] > energy[i - 1] && energy[i] > energy[i + 1] {
                if let Some(last) = last_peak_idx {
                    peak_intervals.push((i - last) as f32);
                }
                last_peak_idx = Some(i);
            }
        }

        if peak_intervals.len() < 2 {
            return 0.0;
        }

        let mean: f32 = peak_intervals.iter().sum::<f32>() / peak_intervals.len() as f32;
        if mean < 1e-10 {
            return 0.0;
        }

        let variance: f32 = peak_intervals
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / peak_intervals.len() as f32;

        variance.sqrt() / mean // CV = std / mean
    }

    /// Classify the linguistic profile based on acoustic features
    fn classify_profile(
        &self,
        energy_variance: f32,
        rhythmicity: f32,
        spectral_stability: f32,
        duration_cv: f32,
    ) -> (LinguisticProfile, f32) {
        // Crystallized Song: Low variance, high rhythmicity, stable spectrum
        if energy_variance < self.config.energy_variance_threshold
            && rhythmicity > self.config.rhythmicity_threshold
            && spectral_stability > 0.7
        {
            let confidence = (rhythmicity + spectral_stability) / 2.0;
            return (LinguisticProfile::CrystallizedSong, confidence);
        }

        // Graded Call: High duration CV, variable energy, low rhythmicity
        if duration_cv > self.config.duration_cv_threshold && rhythmicity < 0.5 {
            let confidence = duration_cv.min(1.0);
            return (LinguisticProfile::GradedCall, confidence);
        }

        // Cultural Coda: High rhythmicity but with learned variation
        if rhythmicity > 0.7 && spectral_stability > 0.5 && duration_cv < 0.4 {
            return (LinguisticProfile::CulturalCoda, rhythmicity);
        }

        // Mixed: Doesn't fit clear patterns
        (LinguisticProfile::Mixed, 0.5)
    }

    /// Route to appropriate segmentation method based on analysis
    ///
    /// # Default Behavior (v2.1.0)
    ///
    /// NBD is now the **default** method. CPD is only used when:
    /// 1. `forced_mode == ForceCPD`, OR
    /// 2. `forced_mode == Auto` AND high-confidence crystallized song detected
    ///
    /// This prevents EBD from fragmenting graded signals incorrectly.
    fn route_and_segment(
        &mut self,
        audio: &[f32],
        analysis: &PreliminaryAnalysis,
    ) -> (Vec<PhraseBoundary>, SegmentationMethod) {
        // Check forced mode first
        let method = match self.config.forced_mode {
            ForcedSegmentationMode::ForceNBD => SegmentationMethod::NBD,
            ForcedSegmentationMode::ForceCPD => SegmentationMethod::CPD,
            ForcedSegmentationMode::ForceHybrid => SegmentationMethod::Hybrid,
            ForcedSegmentationMode::Auto => {
                // Auto mode: Default to NBD, only use CPD for high-confidence crystallized song
                match analysis.profile {
                    LinguisticProfile::CrystallizedSong => {
                        // Only use CPD if confidence is very high
                        if analysis.confidence >= self.config.crystallized_confidence_threshold {
                            SegmentationMethod::CPD
                        } else {
                            // Fall back to NBD for uncertain cases
                            SegmentationMethod::NBD
                        }
                    }
                    LinguisticProfile::CulturalCoda => {
                        // Cultural codas (sperm whale) - still prefer NBD unless high confidence
                        if analysis.confidence >= self.config.crystallized_confidence_threshold {
                            SegmentationMethod::CPD
                        } else {
                            SegmentationMethod::NBD
                        }
                    }
                    LinguisticProfile::GradedCall => SegmentationMethod::NBD,
                    LinguisticProfile::Mixed => {
                        // Mixed now defaults to NBD instead of Hybrid to avoid EBD fragmentation
                        SegmentationMethod::NBD
                    }
                }
            }
        };

        self.segment_with_method(audio, method)
    }

    /// Segment audio using a specific method
    fn segment_with_method(
        &mut self,
        audio: &[f32],
        method: SegmentationMethod,
    ) -> (Vec<PhraseBoundary>, SegmentationMethod) {
        match method {
            SegmentationMethod::CPD => {
                let boundaries = self.segment_cpd(audio);
                (boundaries, method)
            }
            SegmentationMethod::NBD => {
                let boundaries = self.nbd_detector.detect_boundaries(audio);
                (boundaries, method)
            }
            SegmentationMethod::Hybrid => {
                let cpd_boundaries = self.segment_cpd(audio);
                let nbd_boundaries = self.nbd_detector.detect_boundaries(audio);
                let merged = self.merge_boundaries(&cpd_boundaries, &nbd_boundaries);
                (merged, method)
            }
        }
    }

    /// Segment using CPD (energy-based change point detection)
    fn segment_cpd(&mut self, audio: &[f32]) -> Vec<PhraseBoundary> {
        let hop_size = 512;
        let n_frames = audio.len() / hop_size;

        // Compute energy profile
        let energy: Vec<f32> = (0..n_frames)
            .map(|i| {
                let start = i * hop_size;
                let end = (start + hop_size).min(audio.len());
                let frame = &audio[start..end];
                let sum: f32 = frame.iter().map(|x| x * x).sum();
                (sum / frame.len() as f32).sqrt()
            })
            .collect();

        // Find peaks in energy derivative (change points)
        let mut boundaries: Vec<PhraseBoundary> = Vec::new();
        let min_samples =
            (self.config.scan_duration_ms * self.config.sample_rate as f32 / 1000.0) as usize;

        if energy.len() < 5 {
            return boundaries; // Not enough samples to detect boundaries
        }

        for i in 2..energy.len() - 2 {
            let derivative = (energy[i] - energy[i - 1]).abs();

            // Check if this is a local maximum in derivative
            let prev_deriv = (energy[i - 1] - energy[i - 2]).abs();
            let next_deriv = (energy[i + 1] - energy[i]).abs();

            if derivative > prev_deriv && derivative > next_deriv && derivative > 0.1 {
                let time_ms = (i * hop_size) as f32 / self.config.sample_rate as f32 * 1000.0;

                // Debounce
                if let Some(last) = boundaries.last() {
                    if (time_ms - last.time_ms)
                        < min_samples as f32 / self.config.sample_rate as f32 * 1000.0
                    {
                        continue;
                    }
                }

                boundaries.push(PhraseBoundary {
                    time_ms,
                    confidence: derivative.min(1.0),
                    boundary_type: BoundaryType::Hard,
                });
            }
        }

        boundaries
    }

    /// Merge boundaries from two methods (for hybrid approach)
    fn merge_boundaries(
        &self,
        cpd: &[PhraseBoundary],
        nbd: &[PhraseBoundary],
    ) -> Vec<PhraseBoundary> {
        let mut merged: Vec<PhraseBoundary> = Vec::new();
        let merge_threshold_ms = 30.0;

        // Add all CPD boundaries
        for b in cpd {
            merged.push(*b);
        }

        // Add NBD boundaries that don't overlap with CPD
        for b in nbd {
            let is_duplicate = merged
                .iter()
                .any(|m| (m.time_ms - b.time_ms).abs() < merge_threshold_ms);

            if !is_duplicate {
                merged.push(*b);
            }
        }

        // Sort by time
        merged.sort_by(|a, b| {
            a.time_ms
                .partial_cmp(&b.time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        merged
    }

    /// Compute Zipf correlation for quality assessment
    fn compute_zipf_correlation(&self, boundaries: &[PhraseBoundary]) -> f32 {
        if boundaries.len() < 3 {
            return 0.0;
        }

        // Compute durations
        let mut durations: Vec<f32> = Vec::new();
        for i in 1..boundaries.len() {
            durations.push(boundaries[i].time_ms - boundaries[i - 1].time_ms);
        }

        // Add final duration (to end of typical phrase)
        if let Some(last) = boundaries.last() {
            durations.push(200.0); // Assume 200ms final phrase
        }

        if durations.is_empty() {
            return 0.0;
        }

        // Simple Zipf check: rank-frequency correlation
        // Count frequency of similar durations (binned)
        let bin_size = 20.0; // 20ms bins
        let mut freq: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

        for &d in &durations {
            let bin = (d / bin_size).max(1.0) as usize;
            *freq.entry(bin).or_insert(0) += 1;
        }

        // Compute log-log correlation
        let mut freqs: Vec<usize> = freq.values().cloned().collect();
        freqs.sort_by(|a, b| b.cmp(a));

        if freqs.len() < 3 {
            return 0.0;
        }

        // Simple R² approximation
        let n = freqs.len() as f32;
        let sum_x: f32 = (1..=freqs.len()).map(|i| (i as f32).ln()).sum();
        let sum_y: f32 = freqs.iter().map(|&f| (f as f32 + 1.0).ln()).sum();
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for (i, &f) in freqs.iter().enumerate() {
            let x = ((i + 1) as f32).ln();
            let y = (f as f32 + 1.0).ln();
            num += (x - mean_x) * (y - mean_y);
            den_x += (x - mean_x).powi(2);
            den_y += (y - mean_y).powi(2);
        }

        if den_x < 1e-10 || den_y < 1e-10 {
            return 0.0;
        }

        let r = num / (den_x * den_y).sqrt();
        r * r // R²
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_segmenter_creation() {
        let segmenter = SmartSegmenter::new(44100);
        assert_eq!(segmenter.config.sample_rate, 44100);
    }

    #[test]
    fn test_preliminary_analysis_silence() {
        let segmenter = SmartSegmenter::new(44100);
        let silence = vec![0.0f32; 44100]; // 1 second of silence

        let analysis = segmenter.analyze_preliminary(&silence);
        assert!(analysis.energy_variance < 0.01);
    }

    #[test]
    fn test_preliminary_analysis_sine() {
        let segmenter = SmartSegmenter::new(44100);
        let sample_rate = 44100.0f32;
        let sine: Vec<f32> = (0..sample_rate as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin() * 0.5)
            .collect();

        let analysis = segmenter.analyze_preliminary(&sine);
        assert!(analysis.spectral_stability > 0.5);
    }

    #[test]
    fn test_rhythmicity_computation() {
        let segmenter = SmartSegmenter::new(44100);

        // Create rhythmic energy pattern
        let rhythmic: Vec<f32> = (0..100)
            .map(|i| if i % 10 < 5 { 1.0 } else { 0.0 })
            .collect();

        let rhythmicity = segmenter.compute_rhythmicity(&rhythmic);
        assert!(rhythmicity > 0.5);
    }

    #[test]
    fn test_duration_cv_estimation() {
        let segmenter = SmartSegmenter::new(44100);

        // Create energy with regular peaks (low CV)
        let regular_peaks: Vec<f32> = (0..100)
            .map(|i| if i % 10 == 0 { 1.0 } else { 0.1 })
            .collect();

        let cv_regular = segmenter.estimate_duration_cv(&regular_peaks);
        assert!(cv_regular < 0.5);

        // Create energy with irregular peaks (high CV)
        let irregular_peaks: Vec<f32> = (0..100)
            .map(|i| {
                let pos = i % 15; // Irregular pattern
                if pos == 0 || pos == 7 || pos == 12 {
                    1.0
                } else {
                    0.1
                }
            })
            .collect();

        let cv_irregular = segmenter.estimate_duration_cv(&irregular_peaks);
        assert!(cv_irregular > cv_regular);
    }

    #[test]
    fn test_profile_classification() {
        let segmenter = SmartSegmenter::new(44100);

        // Crystallized song: low variance, high rhythmicity
        let (profile1, _) = segmenter.classify_profile(0.05, 0.8, 0.9, 0.2);
        assert_eq!(profile1, LinguisticProfile::CrystallizedSong);

        // Graded call: high CV, low rhythmicity
        let (profile2, _) = segmenter.classify_profile(0.5, 0.3, 0.4, 0.6);
        assert_eq!(profile2, LinguisticProfile::GradedCall);
    }

    #[test]
    fn test_boundary_merging() {
        let segmenter = SmartSegmenter::new(44100);

        let cpd = vec![
            PhraseBoundary {
                time_ms: 100.0,
                confidence: 0.8,
                boundary_type: BoundaryType::Hard,
            },
            PhraseBoundary {
                time_ms: 200.0,
                confidence: 0.7,
                boundary_type: BoundaryType::Hard,
            },
        ];

        let nbd = vec![
            PhraseBoundary {
                time_ms: 105.0,
                confidence: 0.9,
                boundary_type: BoundaryType::Soft,
            },
            PhraseBoundary {
                time_ms: 350.0,
                confidence: 0.6,
                boundary_type: BoundaryType::Soft,
            },
        ];

        let merged = segmenter.merge_boundaries(&cpd, &nbd);

        // Should have 3 boundaries (105ms is duplicate of 100ms)
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_segment_smart_silence() {
        let mut segmenter = SmartSegmenter::new(44100);
        let silence = vec![0.0f32; 44100];

        let result = segmenter.segment_smart(&silence);
        assert!(result.boundaries.is_empty() || result.boundaries.len() <= 2);
    }

    #[test]
    fn test_segmentation_result_serialization() {
        let result = SmartSegmentationResult {
            boundaries: vec![PhraseBoundary {
                time_ms: 100.0,
                confidence: 0.8,
                boundary_type: BoundaryType::Hard,
            }],
            method: SegmentationMethod::CPD,
            analysis: PreliminaryAnalysis {
                energy_variance: 0.1,
                rhythmicity: 0.7,
                spectral_stability: 0.8,
                estimated_duration_cv: 0.25,
                profile: LinguisticProfile::CrystallizedSong,
                confidence: 0.75,
            },
            zipf_r2: Some(0.9),
            processing_time_ms: 5.0,
        };

        let json = serde_json::to_string(&result).unwrap();
        let decoded: SmartSegmentationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.method, SegmentationMethod::CPD);
        assert_eq!(
            decoded.analysis.profile,
            LinguisticProfile::CrystallizedSong
        );
    }
}
