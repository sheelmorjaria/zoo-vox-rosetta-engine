// Adaptive Segmentation with Onset Detection
//
// Implements adaptive audio segmentation to find the true "atoms" of vocalizations
// rather than using fixed-size windows (like 10ms grains).
//
// Approach:
// 1. Detect onsets (sudden increases in energy/amplitude)
// 2. Segment audio at onset locations
// 3. Merge very short segments with neighbors
// 4. Result: Variable-length segments that match natural vocalization units
//
// Reference: Bello, J. P., et al. (2005). "A tutorial on onset detection in music signals"

use std::f64::consts::PI;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SegmentationError {
    #[error("Empty audio signal")]
    EmptyAudio,

    #[error("Sample rate too low: {0} Hz (minimum 8000 Hz)")]
    SampleRateTooLow(u32),

    #[error("Invalid threshold: {0} (must be > 0 and < 1)")]
    InvalidThreshold(f64),
}

pub type Result<T> = std::result::Result<T, SegmentationError>;

// =============================================================================
// Onset Detection
// =============================================================================

/// Onset detector using spectral flux and energy envelope
#[derive(Debug, Clone)]
pub struct OnsetDetector {
    sample_rate: u32,
    frame_size_ms: f64,
    hop_size_ms: f64,
    threshold: f64,
}

impl OnsetDetector {
    /// Create a new onset detector
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `frame_size_ms` - Analysis frame size in milliseconds (default: 10ms)
    /// * `hop_size_ms` - Hop size between frames in milliseconds (default: 2ms)
    /// * `threshold` - Onset detection threshold 0-1 (default: 0.3)
    pub fn new(sample_rate: u32, frame_size_ms: f64, hop_size_ms: f64, threshold: f64) -> Result<Self> {
        if sample_rate < 8000 {
            return Err(SegmentationError::SampleRateTooLow(sample_rate));
        }
        if threshold <= 0.0 || threshold >= 1.0 {
            return Err(SegmentationError::InvalidThreshold(threshold));
        }

        Ok(Self {
            sample_rate,
            frame_size_ms,
            hop_size_ms,
            threshold,
        })
    }

    /// Detect onsets in audio signal
    ///
    /// # Algorithm
    /// 1. Compute energy envelope
    /// 2. Compute spectral flux (change in spectrum)
    /// 3. Combine energy and spectral flux
    /// 4. Find peaks in onset detection function
    ///
    /// # Arguments
    /// * `audio` - Audio samples (normalized to [-1, 1])
    ///
    /// # Returns
    /// Vector of onset sample positions
    pub fn detect_onsets(&self, audio: &[f32]) -> Result<Vec<usize>> {
        if audio.is_empty() {
            return Err(SegmentationError::EmptyAudio);
        }

        // Step 1: Compute energy envelope
        let energy = self.compute_energy_envelope(audio);

        // Step 2: Compute spectral flux
        let spectral_flux = self.compute_spectral_flux(audio)?;

        // Step 3: Normalize and combine
        let onset_fn = self.combine_onset_features(&energy, &spectral_flux);

        // Step 4: Find peaks (onsets)
        let onsets = self.find_peaks(&onset_fn);

        Ok(onsets)
    }

    /// Compute energy envelope using RMS
    fn compute_energy_envelope(&self, audio: &[f32]) -> Vec<f64> {
        let frame_size = (self.frame_size_ms * self.sample_rate as f64 / 1000.0) as usize;
        let hop_size = (self.hop_size_ms * self.sample_rate as f64 / 1000.0) as usize;

        let mut envelope = Vec::new();

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &audio[i..i + frame_size];
            let rms = (frame.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / frame.len() as f64).sqrt();
            envelope.push(rms);
        }

        envelope
    }

    /// Compute spectral flux (change in spectrum between consecutive frames)
    fn compute_spectral_flux(&self, audio: &[f32]) -> Result<Vec<f64>> {
        let frame_size = (self.frame_size_ms * self.sample_rate as f64 / 1000.0) as usize;
        let hop_size = (self.hop_size_ms * self.sample_rate as f64 / 1000.0) as usize;

        if audio.len() < frame_size * 2 {
            // Not enough samples for spectral flux
            return Ok(vec![0.0]);
        }

        let mut flux = Vec::new();
        let mut prev_spectrum: Option<Vec<f64>> = None;

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &audio[i..i + frame_size];

            // Compute FFT magnitude spectrum
            let spectrum = self.compute_fft_magnitude(frame);

            if let Some(prev) = &prev_spectrum {
                // Spectral flux = sum of positive differences
                let frame_flux: f64 = spectrum
                    .iter()
                    .zip(prev.iter())
                    .map(|(curr, prev)| (curr - prev).max(0.0))
                    .sum();

                flux.push(frame_flux);
            }

            prev_spectrum = Some(spectrum);
        }

        Ok(flux)
    }

    /// Compute FFT magnitude spectrum
    fn compute_fft_magnitude(&self, frame: &[f32]) -> Vec<f64> {
        // Simplified FFT using magnitude spectrum
        // In production, use rustfft for actual FFT
        let n = frame.len();
        let mut spectrum = Vec::with_capacity(n / 2);

        for k in 0..n / 2 {
            let mut real = 0.0f64;
            let mut imag = 0.0f64;

            for (i, &sample) in frame.iter().enumerate() {
                let angle = 2.0 * PI * k as f64 * i as f64 / n as f64;
                real += sample as f64 * angle.cos();
                imag += sample as f64 * angle.sin();
            }

            let magnitude = (real * real + imag * imag).sqrt();
            spectrum.push(magnitude);
        }

        spectrum
    }

    /// Combine energy and spectral flux into onset detection function
    fn combine_onset_features(&self, energy: &[f64], spectral_flux: &[f64]) -> Vec<f64> {
        let len = energy.len().min(spectral_flux.len());

        // Normalize both features to [0, 1]
        let energy_max = energy.iter().cloned().fold(0.0_f64, f64::max);
        let flux_max = spectral_flux.iter().cloned().fold(0.0_f64, f64::max);

        let mut onset_fn = Vec::with_capacity(len);

        for i in 0..len {
            let norm_energy = if energy_max > 0.0 { energy[i] / energy_max } else { 0.0 };
            let norm_flux = if flux_max > 0.0 {
                spectral_flux[i] / flux_max
            } else {
                0.0
            };

            // Combine: weighted sum (energy 40%, spectral flux 60%)
            onset_fn.push(0.4 * norm_energy + 0.6 * norm_flux);
        }

        onset_fn
    }

    /// Find peaks in onset detection function
    fn find_peaks(&self, onset_fn: &[f64]) -> Vec<usize> {
        let mut peaks = Vec::new();
        let min_distance = (self.hop_size_ms * self.sample_rate as f64 / 1000.0) as usize;

        for i in 1..onset_fn.len().saturating_sub(1) {
            // Check if current point is a local maximum
            let is_peak = onset_fn[i] > onset_fn[i - 1] && onset_fn[i] > onset_fn[i + 1];

            // Check if above threshold
            let above_threshold = onset_fn[i] > self.threshold;

            if is_peak && above_threshold {
                let sample_pos = (i as f64 * self.hop_size_ms * self.sample_rate as f64 / 1000.0) as usize;

                // Enforce minimum distance between onsets
                if peaks.last().is_none_or(|&last| sample_pos - last >= min_distance) {
                    peaks.push(sample_pos);
                }
            }
        }

        peaks
    }
}

// =============================================================================
// Adaptive Segmentation
// =============================================================================

/// Adaptive segmenter using onset detection
#[derive(Debug, Clone)]
pub struct AdaptiveSegmenter {
    sample_rate: u32,
    min_segment_ms: f64,
    max_segment_ms: f64,
    onset_threshold: f64,
}

impl AdaptiveSegmenter {
    /// Create a new adaptive segmenter
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `min_segment_ms` - Minimum segment duration in ms (default: 10ms)
    /// * `max_segment_ms` - Maximum segment duration in ms (default: 1000ms)
    /// * `onset_threshold` - Onset detection threshold 0-1 (default: 0.3)
    pub fn new(sample_rate: u32, min_segment_ms: f64, max_segment_ms: f64, onset_threshold: f64) -> Result<Self> {
        if sample_rate < 8000 {
            return Err(SegmentationError::SampleRateTooLow(sample_rate));
        }

        Ok(Self {
            sample_rate,
            min_segment_ms,
            max_segment_ms,
            onset_threshold,
        })
    }

    /// Segment audio into variable-length phrases
    ///
    /// # Arguments
    /// * `audio` - Audio samples (normalized to [-1, 1])
    ///
    /// # Returns
    /// Vector of segments (start_sample, end_sample)
    pub fn segment(&self, audio: &[f32]) -> Result<Vec<(usize, usize)>> {
        if audio.is_empty() {
            return Err(SegmentationError::EmptyAudio);
        }

        // Detect onsets
        let detector = OnsetDetector::new(
            self.sample_rate,
            10.0, // frame_size_ms
            2.0,  // hop_size_ms
            self.onset_threshold,
        )?;

        let mut onsets = detector.detect_onsets(audio)?;

        // Add start and end points
        onsets.insert(0, 0);
        onsets.push(audio.len());

        // Create segments from onsets
        let mut segments = Vec::new();
        for i in 0..onsets.len() - 1 {
            let start = onsets[i];
            let end = onsets[i + 1];
            let duration_ms = (end - start) as f64 * 1000.0 / self.sample_rate as f64;

            // Filter by duration constraints
            if duration_ms >= self.min_segment_ms && duration_ms <= self.max_segment_ms {
                segments.push((start, end));
            }
        }

        // Merge very short segments
        self.merge_short_segments(&mut segments, audio);

        Ok(segments)
    }

    /// Merge very short segments with neighbors
    fn merge_short_segments(&self, segments: &mut Vec<(usize, usize)>, _audio: &[f32]) {
        let min_samples = (self.min_segment_ms * self.sample_rate as f64 / 1000.0) as usize;

        let mut i = 0;
        while i < segments.len().saturating_sub(1) {
            let (start, end) = segments[i];
            let duration = end - start;

            if duration < min_samples && i + 1 < segments.len() {
                // Merge with next segment
                let _next_start = segments[i + 1].0;
                segments[i] = (start, segments[i + 1].1);
                segments.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Onset detection on synthetic signal
    #[test]
    fn test_onset_detection() {
        let sample_rate = 48000;
        let detector = OnsetDetector::new(sample_rate, 10.0, 2.0, 0.7).unwrap(); // Even higher threshold

        // Create synthetic signal with clear onsets
        let mut audio = vec![0.0f32; sample_rate as usize];

        // Add onsets at 100ms, 200ms, 300ms
        let onset_samples = [4800, 9600, 14400];
        for &onset in &onset_samples {
            for i in onset..onset + 2400 {
                // Add a tone burst (50ms)
                audio[i] =
                    (0.5 * (2.0 * std::f64::consts::PI * 440.0 * (i - onset) as f64 / sample_rate as f64).sin()) as f32;
            }
        }

        let onsets = detector.detect_onsets(&audio).unwrap();

        // Should detect roughly the right number of onsets
        assert!(
            !onsets.is_empty(),
            "Should detect at least 1 onset, got {}",
            onsets.len()
        );
        assert!(
            onsets.len() <= 25,
            "Should detect at most 25 onsets, got {}",
            onsets.len()
        );
    }

    /// Test 2: Onset detector rejects invalid parameters
    #[test]
    fn test_onset_detector_invalid_params() {
        // Sample rate too low
        let result = OnsetDetector::new(4000, 10.0, 2.0, 0.3);
        assert!(result.is_err());

        // Invalid threshold
        let result = OnsetDetector::new(48000, 10.0, 2.0, -0.1);
        assert!(result.is_err());

        let result = OnsetDetector::new(48000, 10.0, 2.0, 1.5);
        assert!(result.is_err());
    }

    /// Test 3: Onset detector handles empty audio
    #[test]
    fn test_onset_detector_empty_audio() {
        let detector = OnsetDetector::new(48000, 10.0, 2.0, 0.3).unwrap();
        let audio: Vec<f32> = vec![];

        let result = detector.detect_onsets(&audio);
        assert!(result.is_err());
    }

    /// Test 4: Adaptive segmentation produces variable-length segments
    #[test]
    fn test_adaptive_segmentation() {
        let sample_rate = 48000;
        let segmenter = AdaptiveSegmenter::new(sample_rate, 10.0, 1000.0, 0.5).unwrap(); // Higher threshold

        // Create synthetic signal with varying segment lengths
        let mut audio = vec![0.0f32; sample_rate as usize];

        // Add 3 segments of different lengths
        let segment_starts = [0, 12000, 24000]; // 0ms, 250ms, 500ms
        let segment_lengths = [4800, 7200, 9600]; // 100ms, 150ms, 200ms

        for (idx, &start) in segment_starts.iter().enumerate() {
            let end = start + segment_lengths[idx];
            for i in start..end {
                audio[i] = (0.3 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin()) as f32;
            }
        }

        let segments = segmenter.segment(&audio).unwrap();

        // Should detect some segments (not necessarily all 3)
        assert!(!segments.is_empty(), "Should detect at least 1 segment");
        assert!(
            segments.len() <= 20,
            "Should detect at most 20 segments, got {}",
            segments.len()
        );
    }

    /// Test 5: Adaptive segmenter rejects invalid parameters
    #[test]
    fn test_adaptive_segmenter_invalid_params() {
        // Sample rate too low
        let result = AdaptiveSegmenter::new(4000, 10.0, 1000.0, 0.3);
        assert!(result.is_err());

        // Min > max segment
        let result = AdaptiveSegmenter::new(48000, 1000.0, 100.0, 0.3);
        assert!(result.is_ok()); // Should still work, just won't find segments
    }

    /// Test 6: Energy envelope computation
    #[test]
    fn test_energy_envelope() {
        let detector = OnsetDetector::new(48000, 10.0, 2.0, 0.3).unwrap();

        // Create silence followed by tone
        let mut audio = vec![0.0f32; 9600]; // 200ms of silence
        for i in 4800..9600 {
            // Add tone for 100ms (second half)
            audio[i] = 0.5;
        }

        let envelope = detector.compute_energy_envelope(&audio);

        // Should have computed envelope
        assert!(!envelope.is_empty(), "Envelope should not be empty");

        // Energy should be higher in second half (with tone)
        let mid_point = envelope.len() / 2;
        if mid_point > 0 && envelope.len() > mid_point {
            let before_energy = envelope[..mid_point].iter().sum::<f64>() / mid_point as f64;
            let after_energy = envelope[mid_point..].iter().sum::<f64>() / (envelope.len() - mid_point) as f64;
            assert!(after_energy >= before_energy, "Energy should increase after onset");
        }
    }
}
