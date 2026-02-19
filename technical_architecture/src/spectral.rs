// =============================================================================
// Spectral Analysis Module - Dolphin FM Whistle Analysis
// =============================================================================
//
// Analyzes frequency-modulated (FM) signals for dolphin whistle analysis.
// This module is required for species that use frequency contours rather than
// temporal phrase patterns for communication encoding.

use serde::{Deserialize, Serialize};

/// Configuration for frequency contour detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourConfig {
    /// Minimum frequency sweep range in Hz
    pub min_sweep_range: f64,

    /// Minimum contour duration in ms
    pub min_duration_ms: f64,

    /// Number of frequency bins for discretization
    pub frequency_bins: usize,

    /// Number of time bins for discretization
    pub time_bins: usize,
}

impl Default for ContourConfig {
    fn default() -> Self {
        Self {
            min_sweep_range: 500.0,
            min_duration_ms: 100.0,
            frequency_bins: 8,
            time_bins: 10,
        }
    }
}

/// Frequency modulation type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FMType {
    /// Rising sweep (upsweep)
    Rising,

    /// Falling sweep (downsweep)
    Falling,

    /// U-shaped (down then up)
    UShaped,

    /// Inverted U (up then down)
    InvertedU,

    /// Complex (multiple inflections)
    Complex,

    /// Flat (minimal modulation)
    Flat,
}

impl std::fmt::Display for FMType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FMType::Rising => write!(f, "Rising"),
            FMType::Falling => write!(f, "Falling"),
            FMType::UShaped => write!(f, "U-Shaped"),
            FMType::InvertedU => write!(f, "Inverted-U"),
            FMType::Complex => write!(f, "Complex"),
            FMType::Flat => write!(f, "Flat"),
        }
    }
}

/// Features extracted from a frequency contour
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourFeatures {
    /// Start frequency in Hz
    pub f_start: f64,

    /// End frequency in Hz
    pub f_end: f64,

    /// Minimum frequency in Hz
    pub f_min: f64,

    /// Maximum frequency in Hz
    pub f_max: f64,

    /// Frequency range in Hz
    pub f_range: f64,

    /// Duration in ms
    pub duration_ms: f64,

    /// Frequency slope in Hz/ms
    pub slope: f64,

    /// Number of inflection points
    pub inflections: usize,

    /// FM type classification
    pub fm_type: FMType,
}

/// A detected frequency contour
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyContour {
    /// Contour ID
    pub id: usize,

    /// Frequency trajectory (time_ms, frequency_hz)
    pub trajectory: Vec<(f64, f64)>,

    /// Discretized contour shape signature
    pub shape_signature: Vec<i32>,

    /// Contour features
    pub features: ContourFeatures,

    /// Predicted context/meaning
    pub predicted_context: Option<String>,
}

/// Spectral analysis module for FM whistle analysis
pub struct SpectralModule {
    /// Frequency resolution in Hz
    frequency_resolution: f64,

    /// Time resolution in ms
    time_resolution: f64,

    /// Contour detection configuration
    config: ContourConfig,
}

impl SpectralModule {
    /// Create a new spectral module with the given configuration
    pub fn new(config: ContourConfig) -> Self {
        Self {
            frequency_resolution: 10.0,
            time_resolution: 1.0,
            config,
        }
    }

    /// Get frequency resolution
    pub fn frequency_resolution(&self) -> f64 {
        self.frequency_resolution
    }

    /// Get time resolution
    pub fn time_resolution(&self) -> f64 {
        self.time_resolution
    }

    /// Analyze audio to detect FM contours
    pub fn analyze(&self, audio: &[f32], sample_rate: u32) -> Vec<FrequencyContour> {
        if audio.is_empty() {
            return Vec::new();
        }

        // Step 1: Compute spectrogram
        let spectrogram = self.compute_spectrogram(audio, sample_rate);

        // Step 2: Detect frequency contours (ridge detection)
        let ridges = self.detect_contours(&spectrogram, sample_rate);

        // Step 3: Extract features for each contour
        ridges
            .into_iter()
            .enumerate()
            .map(|(id, ridge)| self.create_contour(id, ridge, sample_rate))
            .filter(|c| c.features.f_range >= self.config.min_sweep_range)
            .filter(|c| c.features.duration_ms >= self.config.min_duration_ms)
            .collect()
    }

    /// Compute time-frequency spectrogram
    fn compute_spectrogram(&self, audio: &[f32], sample_rate: u32) -> Vec<Vec<f64>> {
        let window_size = (sample_rate as f64 * 0.01) as usize; // 10ms window
        let hop_size = window_size / 2;

        if window_size == 0 || audio.len() < window_size {
            return Vec::new();
        }

        let n_frames = (audio.len() - window_size) / hop_size + 1;
        let n_freq_bins = window_size / 2 + 1;

        let mut spectrogram = vec![vec![0.0f64; n_freq_bins]; n_frames];

        // Simple DFT for each frame (can be optimized with FFT)
        for frame in 0..n_frames {
            let start = frame * hop_size;
            let end = (start + window_size).min(audio.len());

            for freq_bin in 0..n_freq_bins {
                let mut real = 0.0f64;
                let mut imag = 0.0f64;

                for (i, &sample) in audio[start..end].iter().enumerate() {
                    let phase = 2.0 * std::f64::consts::PI * freq_bin as f64 * i as f64
                        / window_size as f64;
                    real += sample as f64 * phase.cos();
                    imag -= sample as f64 * phase.sin();
                }

                spectrogram[frame][freq_bin] =
                    (real * real + imag * imag).sqrt() / window_size as f64;
            }
        }

        spectrogram
    }

    /// Detect frequency contours using ridge following
    fn detect_contours(
        &self,
        spectrogram: &[Vec<f64>],
        sample_rate: u32,
    ) -> Vec<Vec<(usize, usize)>> {
        if spectrogram.is_empty() {
            return Vec::new();
        }

        let n_frames = spectrogram.len();
        let n_freqs = spectrogram[0].len();

        // Find peak frequency at each time frame
        let mut ridge: Vec<(usize, usize)> = Vec::new();
        let mut in_contour = false;
        let mut contours: Vec<Vec<(usize, usize)>> = Vec::new();

        for frame in 0..n_frames {
            // Find frequency with maximum energy
            let mut max_freq = 0;
            let mut max_val = 0.0f64;

            // Only search relevant frequency range (e.g., 2-30 kHz for dolphins)
            let min_freq_bin = (2000.0 / (sample_rate as f64 / 2.0 / n_freqs as f64)) as usize;
            let max_freq_bin = (30000.0 / (sample_rate as f64 / 2.0 / n_freqs as f64)) as usize;
            let max_freq_bin = max_freq_bin.min(n_freqs - 1);

            for freq in min_freq_bin..=max_freq_bin {
                if spectrogram[frame][freq] > max_val {
                    max_val = spectrogram[frame][freq];
                    max_freq = freq;
                }
            }

            // Threshold for contour detection
            if max_val > 0.001 {
                ridge.push((frame, max_freq));
                in_contour = true;
            } else if in_contour {
                // End of contour
                if ridge.len() >= 5 {
                    contours.push(std::mem::take(&mut ridge));
                }
                ridge.clear();
                in_contour = false;
            }
        }

        // Handle final contour
        if ridge.len() >= 5 {
            contours.push(ridge);
        }

        contours
    }

    /// Create a FrequencyContour from a ridge
    fn create_contour(
        &self,
        id: usize,
        ridge: Vec<(usize, usize)>,
        sample_rate: u32,
    ) -> FrequencyContour {
        let n_freqs = (sample_rate as f64 / 2.0 / self.frequency_resolution) as usize;

        // Convert ridge to frequency trajectory
        let trajectory: Vec<(f64, f64)> = ridge
            .iter()
            .map(|(frame, freq_bin)| {
                let time_ms = *frame as f64 * 5.0; // 5ms per frame (50% overlap of 10ms windows)
                let frequency_hz = *freq_bin as f64 * (sample_rate as f64 / 2.0) / n_freqs as f64;
                (time_ms, frequency_hz)
            })
            .collect();

        // Extract frequencies
        let frequencies: Vec<f64> = trajectory.iter().map(|(_, f)| *f).collect();

        // Extract features
        let features = self.extract_contour_features(&frequencies, sample_rate);

        // Create shape signature
        let shape_signature = self.discretize_contour(&frequencies);

        FrequencyContour {
            id,
            trajectory,
            shape_signature,
            features,
            predicted_context: None,
        }
    }

    /// Extract features from a frequency trajectory
    pub fn extract_contour_features(
        &self,
        frequencies: &[f64],
        _sample_rate: u32,
    ) -> ContourFeatures {
        if frequencies.is_empty() {
            return ContourFeatures {
                f_start: 0.0,
                f_end: 0.0,
                f_min: 0.0,
                f_max: 0.0,
                f_range: 0.0,
                duration_ms: 0.0,
                slope: 0.0,
                inflections: 0,
                fm_type: FMType::Flat,
            };
        }

        let f_start = frequencies[0];
        let f_end = frequencies[frequencies.len() - 1];
        let f_min = frequencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let f_max = frequencies.iter().cloned().fold(0.0f64, f64::max);
        let f_range = f_max - f_min;
        let duration_ms = frequencies.len() as f64 * 5.0; // 5ms per frame

        // Compute slope
        let slope = if duration_ms > 0.0 {
            (f_end - f_start) / duration_ms
        } else {
            0.0
        };

        // Count inflection points
        let inflections = self.count_inflections(frequencies);

        // Classify FM type
        let fm_type = self.classify_fm_type(frequencies);

        ContourFeatures {
            f_start,
            f_end,
            f_min,
            f_max,
            f_range,
            duration_ms,
            slope,
            inflections,
            fm_type,
        }
    }

    /// Count inflection points in frequency trajectory
    fn count_inflections(&self, frequencies: &[f64]) -> usize {
        if frequencies.len() < 3 {
            return 0;
        }

        let mut inflections = 0;
        let mut prev_slope = frequencies[1] - frequencies[0];

        for i in 2..frequencies.len() {
            let current_slope = frequencies[i] - frequencies[i - 1];

            // Sign change indicates inflection
            if (prev_slope > 0.0 && current_slope < 0.0)
                || (prev_slope < 0.0 && current_slope > 0.0)
            {
                inflections += 1;
            }

            prev_slope = current_slope;
        }

        inflections
    }

    /// Classify the FM type of a frequency trajectory
    pub fn classify_fm_type(&self, frequencies: &[f64]) -> FMType {
        if frequencies.len() < 2 {
            return FMType::Flat;
        }

        let f_min = frequencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let f_max = frequencies.iter().cloned().fold(0.0f64, f64::max);
        let f_range = f_max - f_min;

        // Flat if minimal frequency variation
        if f_range < 100.0 {
            return FMType::Flat;
        }

        let f_start = frequencies[0];
        let f_end = frequencies[frequencies.len() - 1];
        let inflections = self.count_inflections(frequencies);

        // No inflections
        if inflections == 0 {
            if f_end > f_start {
                return FMType::Rising;
            } else if f_end < f_start {
                return FMType::Falling;
            }
            return FMType::Flat;
        }

        // One inflection - U-shaped or inverted U
        if inflections == 1 {
            // Check if it goes down then up (U) or up then down (inverted U)
            let mid = frequencies.len() / 2;
            let first_half_slope = frequencies[mid] - frequencies[0];
            let second_half_slope = frequencies[frequencies.len() - 1] - frequencies[mid];

            if first_half_slope < 0.0 && second_half_slope > 0.0 {
                return FMType::UShaped;
            } else if first_half_slope > 0.0 && second_half_slope < 0.0 {
                return FMType::InvertedU;
            }
        }

        // Multiple inflections = complex
        FMType::Complex
    }

    /// Discretize contour into shape signature
    pub fn discretize_contour(&self, frequencies: &[f64]) -> Vec<i32> {
        if frequencies.is_empty() {
            return vec![];
        }

        let f_min = frequencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let f_max = frequencies.iter().cloned().fold(0.0f64, f64::max);
        let f_range = (f_max - f_min).max(1.0);

        let n_bins = self.config.frequency_bins;
        let n_time = self.config.time_bins;

        let mut signature = vec![0i32; n_time];

        // Sample frequencies at regular intervals
        for i in 0..n_time {
            let idx = (i * frequencies.len()) / n_time.max(1);
            if idx < frequencies.len() {
                let normalized = (frequencies[idx] - f_min) / f_range;
                signature[i] = (normalized * (n_bins - 1) as f64).round() as i32;
                signature[i] = signature[i].max(0).min((n_bins - 1) as i32);
            }
        }

        signature
    }
}

impl Default for SpectralModule {
    fn default() -> Self {
        Self::new(ContourConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_module_creation() {
        let config = ContourConfig::default();
        let module = SpectralModule::new(config);

        assert_eq!(module.frequency_resolution(), 10.0);
        assert_eq!(module.time_resolution(), 1.0);
    }

    #[test]
    fn test_fm_type_classification_rising() {
        let module = SpectralModule::default();
        let frequencies = vec![5000.0, 7500.0, 10000.0, 12500.0, 15000.0];

        assert_eq!(module.classify_fm_type(&frequencies), FMType::Rising);
    }

    #[test]
    fn test_fm_type_classification_falling() {
        let module = SpectralModule::default();
        let frequencies = vec![15000.0, 12500.0, 10000.0, 7500.0, 5000.0];

        assert_eq!(module.classify_fm_type(&frequencies), FMType::Falling);
    }

    #[test]
    fn test_fm_type_classification_flat() {
        let module = SpectralModule::default();
        let frequencies = vec![10000.0, 10050.0, 10020.0, 10080.0, 10040.0];

        assert_eq!(module.classify_fm_type(&frequencies), FMType::Flat);
    }
}
