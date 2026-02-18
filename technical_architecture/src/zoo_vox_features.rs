//! 30D/45D Micro-Dynamics Feature Extraction for Zoo Vox Rosetta Engine 2.0
//!
//! Implements complete feature extraction pipeline for vocalization analysis.
//! Supports both 30D and extended 45D feature vectors.

use crate::zoo_vox_data_models::{AcousticFeatures30D, AcousticFeatures45D};

use rustfft::{FftPlanner, FftDirection};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Zoo Vox Rosetta feature extraction error type
#[derive(Debug)]
pub enum FeatureError {
    /// Empty audio buffer
    EmptyAudio,
    /// Audio too short
    AudioTooShort,
    /// Processing error
    ProcessingError(String),
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureError::EmptyAudio => write!(f, "Audio buffer is empty"),
            FeatureError::AudioTooShort => write!(f, "Audio buffer too short for analysis"),
            FeatureError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for FeatureError {}

/// Feature extractor for 30D micro-dynamics analysis
pub struct ZooVoxFeatureExtractor {
    sample_rate: u32,
    fft_planner: FftPlanner<f64>,
}

impl ZooVoxFeatureExtractor {
    /// Create new feature extractor
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Extract all 30D features from audio samples
    pub fn extract(&mut self, audio: &[f64]) -> Result<AcousticFeatures30D, FeatureError> {
        if audio.is_empty() {
            return Err(FeatureError::EmptyAudio);
        }

        if audio.len() < 100 {
            return Err(FeatureError::AudioTooShort);
        }

        // Normalize audio
        let audio = self.normalize(audio);

        let mut features = AcousticFeatures30D::new();

        // === FUNDAMENTAL FEATURES (3) ===
        features.mean_f0_hz = self.estimate_f0(&audio);
        features.duration_ms = (audio.len() as f64 / self.sample_rate as f64) * 1000.0;
        features.f0_range_hz = self.estimate_f0_range(&audio);

        // === GRIT FACTORS (3) ===
        features.harmonic_to_noise_ratio = self.compute_hnr(&audio);
        features.spectral_flatness = self.compute_spectral_flatness(&audio);
        features.harmonicity = self.compute_harmonicity(&audio);

        // === MOTION FACTORS (7) ===
        let envelope = self.compute_envelope(&audio);
        features.attack_time_ms = self.compute_attack_time(&envelope);
        features.decay_time_ms = self.compute_decay_time(&envelope);
        features.sustain_level = self.compute_sustain_level(&envelope);
        let (vibrato_rate, vibrato_depth) = self.compute_vibrato(&audio);
        features.vibrato_rate_hz = vibrato_rate;
        features.vibrato_depth = vibrato_depth;
        features.jitter = self.compute_jitter(&audio);
        features.shimmer = self.compute_shimmer(&audio);

        // === FINGERPRINT FACTORS (14) ===
        let mfccs = self.compute_mfccs(&audio, 13);
        features.mfcc_1 = mfccs.get(0).copied().unwrap_or(0.0);
        features.mfcc_2 = mfccs.get(1).copied().unwrap_or(0.0);
        features.mfcc_3 = mfccs.get(2).copied().unwrap_or(0.0);
        features.mfcc_4 = mfccs.get(3).copied().unwrap_or(0.0);
        features.mfcc_5 = mfccs.get(4).copied().unwrap_or(0.0);
        features.mfcc_6 = mfccs.get(5).copied().unwrap_or(0.0);
        features.mfcc_7 = mfccs.get(6).copied().unwrap_or(0.0);
        features.mfcc_8 = mfccs.get(7).copied().unwrap_or(0.0);
        features.mfcc_9 = mfccs.get(8).copied().unwrap_or(0.0);
        features.mfcc_10 = mfccs.get(9).copied().unwrap_or(0.0);
        features.mfcc_11 = mfccs.get(10).copied().unwrap_or(0.0);
        features.mfcc_12 = mfccs.get(11).copied().unwrap_or(0.0);
        features.mfcc_13 = mfccs.get(12).copied().unwrap_or(0.0);
        features.spectral_flux = self.compute_spectral_flux(&audio);

        // === RHYTHM FACTORS (3) ===
        let (ici, onset_rate) = self.compute_rhythm_features(&audio);
        features.median_ici_ms = ici;
        features.onset_rate_hz = onset_rate;
        features.ici_coefficient_of_variation = self.compute_ici_cv(&audio);

        Ok(features)
    }

    /// Extract all 45D features from audio samples (30D + 15 new dimensions)
    pub fn extract_45d(&mut self, audio: &[f64]) -> Result<AcousticFeatures45D, FeatureError> {
        if audio.is_empty() {
            return Err(FeatureError::EmptyAudio);
        }

        if audio.len() < 100 {
            return Err(FeatureError::AudioTooShort);
        }

        // Normalize audio
        let audio = self.normalize(audio);

        let mut features = AcousticFeatures45D::new();

        // === FUNDAMENTAL FEATURES (3) ===
        features.mean_f0_hz = self.estimate_f0(&audio);
        features.duration_ms = (audio.len() as f64 / self.sample_rate as f64) * 1000.0;
        features.f0_range_hz = self.estimate_f0_range(&audio);

        // === GRIT FACTORS (3) ===
        features.harmonic_to_noise_ratio = self.compute_hnr(&audio);
        features.spectral_flatness = self.compute_spectral_flatness(&audio);
        features.harmonicity = self.compute_harmonicity(&audio);

        // === MOTION FACTORS (7) ===
        let envelope = self.compute_envelope(&audio);
        features.attack_time_ms = self.compute_attack_time(&envelope);
        features.decay_time_ms = self.compute_decay_time(&envelope);
        features.sustain_level = self.compute_sustain_level(&envelope);
        let (vibrato_rate, vibrato_depth) = self.compute_vibrato(&audio);
        features.vibrato_rate_hz = vibrato_rate;
        features.vibrato_depth = vibrato_depth;
        features.jitter = self.compute_jitter(&audio);
        features.shimmer = self.compute_shimmer(&audio);

        // === FINGERPRINT FACTORS (14) ===
        let mfccs = self.compute_mfccs(&audio, 13);
        features.mfcc_1 = mfccs.get(0).copied().unwrap_or(0.0);
        features.mfcc_2 = mfccs.get(1).copied().unwrap_or(0.0);
        features.mfcc_3 = mfccs.get(2).copied().unwrap_or(0.0);
        features.mfcc_4 = mfccs.get(3).copied().unwrap_or(0.0);
        features.mfcc_5 = mfccs.get(4).copied().unwrap_or(0.0);
        features.mfcc_6 = mfccs.get(5).copied().unwrap_or(0.0);
        features.mfcc_7 = mfccs.get(6).copied().unwrap_or(0.0);
        features.mfcc_8 = mfccs.get(7).copied().unwrap_or(0.0);
        features.mfcc_9 = mfccs.get(8).copied().unwrap_or(0.0);
        features.mfcc_10 = mfccs.get(9).copied().unwrap_or(0.0);
        features.mfcc_11 = mfccs.get(10).copied().unwrap_or(0.0);
        features.mfcc_12 = mfccs.get(11).copied().unwrap_or(0.0);
        features.mfcc_13 = mfccs.get(12).copied().unwrap_or(0.0);
        features.spectral_flux = self.compute_spectral_flux(&audio);

        // === RHYTHM FACTORS (3) ===
        let (ici, onset_rate) = self.compute_rhythm_features(&audio);
        features.median_ici_ms = ici;
        features.onset_rate_hz = onset_rate;
        features.ici_coefficient_of_variation = self.compute_ici_cv(&audio);

        // === RESONANCE FACTORS (6) - NEW ===
        let formants = self.compute_formants(&audio);
        features.formant_1_hz = formants.get(0).map(|(f, _)| *f).unwrap_or(0.0);
        features.formant_2_hz = formants.get(1).map(|(f, _)| *f).unwrap_or(0.0);
        features.formant_3_hz = formants.get(2).map(|(f, _)| *f).unwrap_or(0.0);
        features.formant_1_bandwidth = formants.get(0).map(|(_, bw)| *bw).unwrap_or(0.0);
        features.formant_2_bandwidth = formants.get(1).map(|(_, bw)| *bw).unwrap_or(0.0);
        features.formant_dispersion = self.compute_formant_dispersion(&formants);

        // === SPECTRAL SHAPE FACTORS (4) - NEW ===
        let spectrum = self.compute_spectrum(&audio);
        features.spectral_centroid = self.compute_spectral_centroid(&spectrum);
        features.spectral_spread = self.compute_spectral_spread(&spectrum, features.spectral_centroid);
        features.spectral_skewness = self.compute_spectral_skewness(&spectrum, features.spectral_centroid, features.spectral_spread);
        features.spectral_kurtosis = self.compute_spectral_kurtosis(&spectrum, features.spectral_centroid, features.spectral_spread);

        // === MODULATION FACTORS (3) - NEW ===
        features.spectral_tilt = self.compute_spectral_tilt(&spectrum);
        features.fm_slope_hz_per_sec = self.compute_fm_slope(&audio);
        features.am_depth = self.compute_am_depth(&audio);

        // === NON-LINEAR FACTORS (2) - NEW ===
        features.subharmonic_ratio = self.compute_subharmonic_ratio(&audio);
        features.spectral_entropy = self.compute_spectral_entropy(&spectrum);

        Ok(features)
    }

    // ========================================================================
    // NORMALIZATION
    // ========================================================================

    fn normalize(&self, audio: &[f64]) -> Vec<f64> {
        let max_val = audio.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        if max_val > 0.0 {
            audio.iter().map(|x| x / max_val).collect()
        } else {
            audio.to_vec()
        }
    }

    // ========================================================================
    // FUNDAMENTAL FEATURES
    // ========================================================================

    /// Estimate mean fundamental frequency using autocorrelation
    fn estimate_f0(&self, audio: &[f64]) -> f64 {
        if audio.len() < 100 {
            return 0.0;
        }

        // Autocorrelation-based F0 estimation
        let min_lag = (self.sample_rate as f64 / 22000.0).floor() as usize; // Max freq ~22kHz
        let max_lag = (self.sample_rate as f64 / 100.0).floor() as usize;   // Min freq ~100Hz

        let max_lag = max_lag.min(audio.len() - 1);
        let min_lag = min_lag.max(1);

        if max_lag <= min_lag {
            return 0.0;
        }

        // Compute autocorrelation
        let mut best_lag = 0;
        let mut best_corr = 0.0;

        for lag in min_lag..max_lag {
            let corr: f64 = audio.iter()
                .take(audio.len() - lag)
                .zip(audio.iter().skip(lag))
                .map(|(a, b)| a * b)
                .sum();

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        if best_lag > 0 {
            self.sample_rate as f64 / best_lag as f64
        } else {
            0.0
        }
    }

    /// Estimate F0 variation range
    fn estimate_f0_range(&self, audio: &[f64]) -> f64 {
        let frame_size = (self.sample_rate as f64 * 0.01) as usize; // 10ms frames
        if frame_size == 0 {
            return 0.0;
        }

        let mut f0_values: Vec<f64> = Vec::new();

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(frame_size) {
            let frame = &audio[i..i + frame_size];
            let f0 = self.estimate_f0(frame);
            if f0 > 0.0 {
                f0_values.push(f0);
            }
        }

        if f0_values.len() > 1 {
            let min_f0 = f0_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_f0 = f0_values.iter().fold(0.0_f64, |a, &b| a.max(b));
            max_f0 - min_f0
        } else {
            0.0
        }
    }

    // ========================================================================
    // GRIT FACTORS
    // ========================================================================

    /// Compute harmonic-to-noise ratio in dB
    fn compute_hnr(&self, audio: &[f64]) -> f64 {
        if audio.len() < 100 {
            return 0.0;
        }

        // Autocorrelation at lag 0 is total energy
        let total_energy: f64 = audio.iter().map(|x| x * x).sum();

        // Find harmonic energy at F0 period
        let f0 = self.estimate_f0(audio);
        if f0 > 0.0 && f0 < self.sample_rate as f64 / 2.0 {
            let period = (self.sample_rate as f64 / f0) as usize;
            if period < audio.len() {
                let harmonic_energy: f64 = audio.iter()
                    .take(audio.len() - period)
                    .zip(audio.iter().skip(period))
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    .abs();

                let noise_energy = total_energy - harmonic_energy;
                if noise_energy > 0.0 && harmonic_energy > 0.0 {
                    return 10.0 * (harmonic_energy / noise_energy).log10();
                }
            }
        }

        0.0
    }

    /// Compute Wiener entropy (spectral flatness)
    fn compute_spectral_flatness(&mut self, audio: &[f64]) -> f64 {
        if audio.len() < 256 {
            return 0.0;
        }

        let spectrum = self.compute_spectrum(audio);
        if spectrum.is_empty() {
            return 0.0;
        }

        // Geometric mean / arithmetic mean
        let log_sum: f64 = spectrum.iter().map(|x| (x + 1e-10).ln()).sum();
        let geometric_mean = (log_sum / spectrum.len() as f64).exp();
        let arithmetic_mean: f64 = spectrum.iter().sum::<f64>() / spectrum.len() as f64;

        if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }

    /// Compute harmonic coherence (0-1)
    fn compute_harmonicity(&self, audio: &[f64]) -> f64 {
        let hnr = self.compute_hnr(audio);
        // Map HNR from dB to 0-1 scale
        // HNR of 20dB ≈ 0.9 harmonicity
        (1.0 - (-hnr / 10.0).exp()).clamp(0.0, 1.0)
    }

    // ========================================================================
    // MOTION FACTORS
    // ========================================================================

    /// Compute amplitude envelope
    fn compute_envelope(&self, audio: &[f64]) -> Vec<f64> {
        // Hilbert-based envelope approximation
        let analytic: Vec<f64> = audio.iter().map(|x| x.abs()).collect();

        // Smooth with lowpass (moving average)
        let kernel_size = (self.sample_rate as f64 * 0.005) as usize; // 5ms
        if kernel_size < 2 {
            return analytic;
        }

        let mut envelope = Vec::with_capacity(analytic.len());
        for i in 0..analytic.len() {
            let start = i.saturating_sub(kernel_size / 2);
            let end = (i + kernel_size / 2).min(analytic.len());
            let avg: f64 = analytic[start..end].iter().sum::<f64>() / (end - start) as f64;
            envelope.push(avg);
        }

        envelope
    }

    /// Compute attack time in ms
    fn compute_attack_time(&self, envelope: &[f64]) -> f64 {
        if envelope.len() < 10 {
            return 0.0;
        }

        let max_val = envelope.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_val == 0.0 {
            return 0.0;
        }

        let threshold_10 = 0.1 * max_val;
        let threshold_90 = 0.9 * max_val;

        let above_10 = envelope.iter().position(|&x| x > threshold_10);
        let above_90 = envelope.iter().position(|&x| x > threshold_90);

        if let (Some(idx_10), Some(idx_90)) = (above_10, above_90) {
            let samples = idx_90.saturating_sub(idx_10);
            samples as f64 / self.sample_rate as f64 * 1000.0
        } else {
            0.0
        }
    }

    /// Compute decay time in ms
    fn compute_decay_time(&self, envelope: &[f64]) -> f64 {
        if envelope.len() < 10 {
            return 0.0;
        }

        let max_val = envelope.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_val == 0.0 {
            return 0.0;
        }

        let max_idx = envelope.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        if let Some(max_idx) = max_idx {
            let threshold_10 = 0.1 * max_val;

            for (i, &val) in envelope[max_idx..].iter().enumerate() {
                if val < threshold_10 {
                    return i as f64 / self.sample_rate as f64 * 1000.0;
                }
            }
        }

        0.0
    }

    /// Compute sustain level (0-1)
    fn compute_sustain_level(&self, envelope: &[f64]) -> f64 {
        if envelope.len() < 10 {
            return 0.0;
        }

        let max_val = envelope.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_val == 0.0 {
            return 0.0;
        }

        let max_idx = envelope.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        if let Some(max_idx) = max_idx {
            let start_sustain = max_idx / 4;
            let end_sustain = max_idx;

            if end_sustain > start_sustain {
                let sustain_region = &envelope[start_sustain..end_sustain];
                let median = self.median(sustain_region);
                return median / max_val;
            }
        }

        0.0
    }

    /// Compute vibrato rate (Hz) and depth (semitones)
    fn compute_vibrato(&mut self, audio: &[f64]) -> (f64, f64) {
        let frame_size = (self.sample_rate as f64 * 0.01) as usize;
        if frame_size == 0 {
            return (0.0, 0.0);
        }

        let mut f0_track: Vec<f64> = Vec::new();

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(frame_size) {
            let frame = &audio[i..i + frame_size];
            let f0 = self.estimate_f0(frame);
            if f0 > 0.0 {
                f0_track.push(f0);
            }
        }

        if f0_track.len() < 10 {
            return (0.0, 0.0);
        }

        let mean_f0: f64 = f0_track.iter().sum::<f64>() / f0_track.len() as f64;

        // Look for periodic variation (4-12 Hz typical vibrato)
        // FFT of F0 track
        let n = f0_track.len();
        let n_fft = n.next_power_of_two();
        let mut fft_input: Vec<Complex<f64>> = f0_track.iter()
            .map(|&x| Complex::new(x - mean_f0, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(n_fft)
            .collect();

        let fft = self.fft_planner.plan_fft(n_fft, FftDirection::Forward);
        fft.process(&mut fft_input);

        // Find peak in vibrato range (4-12 Hz)
        let frame_rate = self.sample_rate as f64 / frame_size as f64;
        let vibrato_range_min = (4.0 * n_fft as f64 / frame_rate) as usize;
        let vibrato_range_max = (12.0 * n_fft as f64 / frame_rate) as usize;

        let mut vibrato_rate = 0.0;
        let mut max_magnitude = 0.0;

        for (i, c) in fft_input.iter().enumerate() {
            if i >= vibrato_range_min && i <= vibrato_range_max {
                let magnitude = (c.re * c.re + c.im * c.im).sqrt();
                if magnitude > max_magnitude {
                    max_magnitude = magnitude;
                    vibrato_rate = i as f64 * frame_rate / n_fft as f64;
                }
            }
        }

        // Compute depth in semitones
        let f0_min = f0_track.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let f0_max = f0_track.iter().fold(0.0_f64, |a, &b| a.max(b));
        let vibrato_depth = if f0_min > 0.0 {
            12.0 * (f0_max / f0_min).log2()
        } else {
            0.0
        };

        (vibrato_rate, vibrato_depth.abs())
    }

    /// Compute frequency perturbation (jitter)
    fn compute_jitter(&self, audio: &[f64]) -> f64 {
        let frame_size = (self.sample_rate as f64 * 0.01) as usize;
        if frame_size == 0 {
            return 0.0;
        }

        let mut f0_track: Vec<f64> = Vec::new();

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(frame_size) {
            let frame = &audio[i..i + frame_size];
            let f0 = self.estimate_f0(frame);
            if f0 > 0.0 {
                f0_track.push(f0);
            }
        }

        if f0_track.len() < 3 {
            return 0.0;
        }

        let mean_f0: f64 = f0_track.iter().sum::<f64>() / f0_track.len() as f64;
        if mean_f0 == 0.0 {
            return 0.0;
        }

        let diffs: f64 = f0_track.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>();

        (diffs / (f0_track.len() - 1) as f64) / mean_f0
    }

    /// Compute amplitude perturbation (shimmer)
    fn compute_shimmer(&self, audio: &[f64]) -> f64 {
        let envelope = self.compute_envelope(audio);
        let frame_size = (self.sample_rate as f64 * 0.01) as usize;
        if frame_size == 0 {
            return 0.0;
        }

        let mut amp_track: Vec<f64> = Vec::new();

        for i in (0..envelope.len().saturating_sub(frame_size)).step_by(frame_size) {
            let avg: f64 = envelope[i..i + frame_size].iter().sum::<f64>() / frame_size as f64;
            amp_track.push(avg);
        }

        if amp_track.len() < 3 {
            return 0.0;
        }

        let mean_amp: f64 = amp_track.iter().sum::<f64>() / amp_track.len() as f64;
        if mean_amp == 0.0 {
            return 0.0;
        }

        let diffs: f64 = amp_track.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>();

        (diffs / (amp_track.len() - 1) as f64) / mean_amp
    }

    // ========================================================================
    // FINGERPRINT FACTORS
    // ========================================================================

    /// Compute MFCCs
    fn compute_mfccs(&mut self, audio: &[f64], n_mfcc: usize) -> Vec<f64> {
        let n_fft = 2048.min(audio.len()).next_power_of_two();

        // Compute spectrum
        let spectrum = self.compute_spectrum_padded(audio, n_fft);

        // Create mel filterbank
        let n_filters = 26;
        let mel_filters = self.create_mel_filterbank(n_fft, n_filters);

        // Apply mel filterbank
        let mel_spectrum: Vec<f64> = mel_filters.iter()
            .map(|filter| {
                filter.iter()
                    .zip(spectrum.iter())
                    .map(|(w, s)| w * s)
                    .sum()
            })
            .collect();

        // Log and DCT
        let log_mel: Vec<f64> = mel_spectrum.iter()
            .map(|&x| (x + 1e-10).ln())
            .collect();

        // DCT-II for MFCCs
        let mut mfccs = Vec::with_capacity(n_mfcc);
        for i in 0..n_mfcc {
            let mut sum = 0.0;
            for (j, &x) in log_mel.iter().enumerate() {
                sum += x * (PI * i as f64 * (j as f64 + 0.5) / log_mel.len() as f64).cos();
            }
            mfccs.push(sum);
        }

        mfccs
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(&self, n_fft: usize, n_filters: usize) -> Vec<Vec<f64>> {
        let low_freq = 0.0;
        let high_freq = self.sample_rate as f64 / 2.0;

        // Hz to mel conversion
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);

        let mel_low = hz_to_mel(low_freq);
        let mel_high = hz_to_mel(high_freq);

        let mel_points: Vec<f64> = (0..=n_filters + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_filters + 1) as f64)
            .collect();

        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points.iter()
            .map(|&hz| ((n_fft as f64 + 1.0) * hz / self.sample_rate as f64).floor() as usize)
            .collect();

        let n_bins = n_fft / 2 + 1;
        let mut filters = Vec::with_capacity(n_filters);

        for i in 0..n_filters {
            let mut filter = vec![0.0; n_bins];
            let bin_left = bin_points[i];
            let bin_center = bin_points[i + 1];
            let bin_right = bin_points[i + 2];

            // Rising edge
            for j in bin_left..bin_center.min(n_bins) {
                if bin_center > bin_left {
                    filter[j] = (j - bin_left) as f64 / (bin_center - bin_left) as f64;
                }
            }

            // Falling edge
            for j in bin_center..bin_right.min(n_bins) {
                if bin_right > bin_center {
                    filter[j] = (bin_right - j) as f64 / (bin_right - bin_center) as f64;
                }
            }

            filters.push(filter);
        }

        filters
    }

    /// Compute spectral flux
    fn compute_spectral_flux(&mut self, audio: &[f64]) -> f64 {
        let frame_size = 1024;
        let hop_size = 512;

        if audio.len() < frame_size {
            return 0.0;
        }

        let mut spectra: Vec<Vec<f64>> = Vec::new();

        for i in (0..audio.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &audio[i..i + frame_size];
            spectra.push(self.compute_spectrum(frame));
        }

        if spectra.len() < 2 {
            return 0.0;
        }

        let mut flux = 0.0;
        for i in 1..spectra.len() {
            for (s1, s2) in spectra[i - 1].iter().zip(spectra[i].iter()) {
                let diff = s2 - s1;
                if diff > 0.0 {
                    flux += diff * diff;
                }
            }
        }

        flux / spectra.len() as f64
    }

    // ========================================================================
    // RHYTHM FACTORS
    // ========================================================================

    /// Compute rhythm features (ICI, onset rate)
    fn compute_rhythm_features(&self, audio: &[f64]) -> (f64, f64) {
        let envelope = self.compute_envelope(audio);

        // Find peaks in derivative
        let diff: Vec<f64> = envelope.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let threshold = if diff.is_empty() {
            0.0
        } else {
            diff.iter().sum::<f64>() / diff.len() as f64 +
                diff.iter().map(|x| x * x).sum::<f64>().sqrt() / diff.len() as f64
        };

        let mut onsets: Vec<usize> = Vec::new();
        for i in 1..diff.len() {
            if diff[i] > threshold && diff[i - 1] <= threshold {
                onsets.push(i);
            }
        }

        if onsets.len() < 2 {
            return (0.0, 0.0);
        }

        // Inter-onset intervals
        let iois: Vec<f64> = onsets.windows(2)
            .map(|w| (w[1] - w[0]) as f64 / self.sample_rate as f64 * 1000.0)
            .collect();

        let median_ici = self.median(&iois);
        let onset_rate = onsets.len() as f64 / (audio.len() as f64 / self.sample_rate as f64);

        (median_ici, onset_rate)
    }

    /// Compute ICI coefficient of variation
    fn compute_ici_cv(&self, audio: &[f64]) -> f64 {
        let envelope = self.compute_envelope(audio);

        let diff: Vec<f64> = envelope.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let threshold = if diff.is_empty() {
            0.0
        } else {
            diff.iter().sum::<f64>() / diff.len() as f64 +
                diff.iter().map(|x| x * x).sum::<f64>().sqrt() / diff.len() as f64
        };

        let mut onsets: Vec<usize> = Vec::new();
        for i in 1..diff.len() {
            if diff[i] > threshold && diff[i - 1] <= threshold {
                onsets.push(i);
            }
        }

        if onsets.len() < 3 {
            return 0.0;
        }

        let iois: Vec<f64> = onsets.windows(2)
            .map(|w| (w[1] - w[0]) as f64 / self.sample_rate as f64 * 1000.0)
            .collect();

        let mean = iois.iter().sum::<f64>() / iois.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }

        let variance: f64 = iois.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / iois.len() as f64;

        variance.sqrt() / mean
    }

    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================

    /// Compute magnitude spectrum
    fn compute_spectrum(&mut self, audio: &[f64]) -> Vec<f64> {
        let n_fft = audio.len().min(2048);
        self.compute_spectrum_padded(audio, n_fft.next_power_of_two())
    }

    /// Compute magnitude spectrum with padding
    fn compute_spectrum_padded(&mut self, audio: &[f64], n_fft: usize) -> Vec<f64> {
        let n_fft = n_fft.max(2);

        // Prepare FFT input
        let mut fft_input: Vec<Complex<f64>> = audio.iter()
            .take(n_fft)
            .map(|&x| Complex::new(x, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(n_fft)
            .collect();

        let fft = self.fft_planner.plan_fft(n_fft, FftDirection::Forward);
        fft.process(&mut fft_input);

        // Return magnitude spectrum (only positive frequencies)
        fft_input[..n_fft / 2 + 1]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect()
    }

    /// Compute median
    fn median(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    // ========================================================================
    // RESONANCE FACTORS (6 features) - NEW for 45D
    // ========================================================================

    /// Compute formant frequencies and bandwidths using LPC
    /// Returns Vec<(frequency_hz, bandwidth_hz)>
    fn compute_formants(&mut self, audio: &[f64]) -> Vec<(f64, f64)> {
        if audio.len() < 512 {
            return Vec::new();
        }

        // Use LPC (Linear Predictive Coding) to estimate formants
        let lpc_order = 12; // Typical for 3-4 formants

        // Compute autocorrelation
        let mut autocorr = vec![0.0; lpc_order + 1];
        for i in 0..=lpc_order {
            let sum: f64 = audio.iter()
                .take(audio.len() - i)
                .zip(audio.iter().skip(i))
                .map(|(a, b)| a * b)
                .sum();
            autocorr[i] = sum;
        }

        if autocorr[0] == 0.0 {
            return Vec::new();
        }

        // Levinson-Durbin recursion for LPC coefficients
        let mut lpc = vec![0.0; lpc_order + 1];
        lpc[0] = 1.0;

        let mut error = autocorr[0];
        let mut reflection = vec![0.0; lpc_order + 1];

        for i in 1..=lpc_order {
            let mut sum = 0.0;
            for j in 1..i {
                sum += lpc[j] * autocorr[i - j];
            }

            reflection[i] = (autocorr[i] - sum) / error;
            error *= 1.0 - reflection[i] * reflection[i];

            for j in 1..i {
                lpc[j] -= reflection[i] * lpc[i - j];
            }
            lpc[i] = reflection[i];
        }

        // Find roots of LPC polynomial to get formants
        // Simplified: find peaks in LPC frequency response
        let n_fft = 2048;
        let mut fft_input: Vec<Complex<f64>> = lpc.iter()
            .map(|&x| Complex::new(x, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(n_fft)
            .collect();

        let fft = self.fft_planner.plan_fft(n_fft, FftDirection::Forward);
        fft.process(&mut fft_input);

        // Compute magnitude response (inverse of LPC spectrum)
        let magnitudes: Vec<f64> = fft_input[..n_fft / 2 + 1]
            .iter()
            .map(|c| 1.0 / ((c.re * c.re + c.im * c.im).sqrt() + 1e-10))
            .collect();

        // Find peaks (formants)
        let mut formants: Vec<(f64, f64)> = Vec::new();
        let freq_per_bin = self.sample_rate as f64 / n_fft as f64;

        for i in 2..magnitudes.len() - 2 {
            // Check if this is a local maximum
            if magnitudes[i] > magnitudes[i - 1]
                && magnitudes[i] > magnitudes[i + 1]
                && magnitudes[i] > magnitudes[i - 2]
                && magnitudes[i] > magnitudes[i + 2]
            {
                let freq = i as f64 * freq_per_bin;

                // Only accept formants in valid range (100-8000 Hz)
                if freq > 100.0 && freq < 8000.0 {
                    // Estimate bandwidth from peak width (3dB points)
                    let peak_val = magnitudes[i];
                    let threshold = peak_val / 2.0_f64.sqrt(); // -3dB

                    // Find left 3dB point
                    let mut left_idx = i;
                    while left_idx > 0 && magnitudes[left_idx] > threshold {
                        left_idx -= 1;
                    }

                    // Find right 3dB point
                    let mut right_idx = i;
                    while right_idx < magnitudes.len() - 1 && magnitudes[right_idx] > threshold {
                        right_idx += 1;
                    }

                    let bandwidth = (right_idx - left_idx) as f64 * freq_per_bin;
                    formants.push((freq, bandwidth));
                }
            }
        }

        // Sort by frequency and take first 3
        formants.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        formants.truncate(3);

        formants
    }

    /// Compute formant dispersion (average spacing between formants)
    fn compute_formant_dispersion(&self, formants: &[(f64, f64)]) -> f64 {
        if formants.len() < 2 {
            return 0.0;
        }

        let total_spacing: f64 = formants.windows(2)
            .map(|w| (w[1].0 - w[0].0).abs())
            .sum();

        total_spacing / (formants.len() - 1) as f64
    }

    // ========================================================================
    // SPECTRAL SHAPE FACTORS (4 features) - NEW for 45D
    // ========================================================================

    /// Compute spectral centroid (center of mass of spectrum)
    fn compute_spectral_centroid(&self, spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() {
            return 0.0;
        }

        let freq_per_bin = self.sample_rate as f64 / (2.0 * (spectrum.len() - 1) as f64);

        let sum_weighted: f64 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| i as f64 * freq_per_bin * mag)
            .sum();

        let sum_mag: f64 = spectrum.iter().sum();

        if sum_mag > 0.0 {
            sum_weighted / sum_mag
        } else {
            0.0
        }
    }

    /// Compute spectral spread (standard deviation around centroid)
    fn compute_spectral_spread(&self, spectrum: &[f64], centroid: f64) -> f64 {
        if spectrum.is_empty() || centroid == 0.0 {
            return 0.0;
        }

        let freq_per_bin = self.sample_rate as f64 / (2.0 * (spectrum.len() - 1) as f64);

        let sum_mag: f64 = spectrum.iter().sum();

        if sum_mag == 0.0 {
            return 0.0;
        }

        let variance: f64 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| {
                let freq = i as f64 * freq_per_bin;
                mag * (freq - centroid).powi(2)
            })
            .sum::<f64>() / sum_mag;

        variance.sqrt()
    }

    /// Compute spectral skewness (asymmetry of spectrum)
    fn compute_spectral_skewness(&self, spectrum: &[f64], centroid: f64, spread: f64) -> f64 {
        if spectrum.is_empty() || spread == 0.0 {
            return 0.0;
        }

        let freq_per_bin = self.sample_rate as f64 / (2.0 * (spectrum.len() - 1) as f64);
        let sum_mag: f64 = spectrum.iter().sum();

        if sum_mag == 0.0 {
            return 0.0;
        }

        let skewness: f64 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| {
                let freq = i as f64 * freq_per_bin;
                mag * ((freq - centroid) / spread).powi(3)
            })
            .sum::<f64>() / sum_mag;

        skewness
    }

    /// Compute spectral kurtosis (peakedness of spectrum)
    fn compute_spectral_kurtosis(&self, spectrum: &[f64], centroid: f64, spread: f64) -> f64 {
        if spectrum.is_empty() || spread == 0.0 {
            return 0.0;
        }

        let freq_per_bin = self.sample_rate as f64 / (2.0 * (spectrum.len() - 1) as f64);
        let sum_mag: f64 = spectrum.iter().sum();

        if sum_mag == 0.0 {
            return 0.0;
        }

        let kurtosis: f64 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| {
                let freq = i as f64 * freq_per_bin;
                mag * ((freq - centroid) / spread).powi(4)
            })
            .sum::<f64>() / sum_mag;

        // Excess kurtosis (subtract 3 for normal distribution)
        kurtosis - 3.0
    }

    // ========================================================================
    // MODULATION FACTORS (3 features) - NEW for 45D
    // ========================================================================

    /// Compute spectral tilt (slope of spectral envelope in dB/octave)
    fn compute_spectral_tilt(&self, spectrum: &[f64]) -> f64 {
        if spectrum.len() < 8 {
            return 0.0;
        }

        let freq_per_bin = self.sample_rate as f64 / (2.0 * (spectrum.len() - 1) as f64);

        // Compute spectral envelope by averaging in octave bands
        let mut octave_energies: Vec<f64> = Vec::new();
        let mut octave_freqs: Vec<f64> = Vec::new();

        let mut low_freq = 100.0;
        while low_freq < self.sample_rate as f64 / 2.0 {
            let high_freq = low_freq * 2.0;

            let low_bin = (low_freq / freq_per_bin) as usize;
            let high_bin = (high_freq / freq_per_bin).min(spectrum.len() as f64) as usize;

            if high_bin > low_bin {
                let energy: f64 = spectrum[low_bin..high_bin].iter().sum::<f64>() / (high_bin - low_bin) as f64;
                if energy > 0.0 {
                    octave_energies.push(10.0 * energy.log10());
                    octave_freqs.push((low_freq + high_freq) / 2.0);
                }
            }

            low_freq = high_freq;
        }

        if octave_energies.len() < 2 {
            return 0.0;
        }

        // Linear regression to find slope (in log2 frequency space)
        let n = octave_energies.len() as f64;
        let sum_x: f64 = octave_freqs.iter().map(|f| f.log2()).sum();
        let sum_y: f64 = octave_energies.iter().sum();
        let sum_xy: f64 = octave_freqs.iter()
            .zip(octave_energies.iter())
            .map(|(f, e)| f.log2() * e)
            .sum();
        let sum_xx: f64 = octave_freqs.iter().map(|f| f.log2().powi(2)).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Compute FM slope (rate of frequency change in Hz/sec)
    fn compute_fm_slope(&mut self, audio: &[f64]) -> f64 {
        let frame_size = (self.sample_rate as f64 * 0.01) as usize; // 10ms frames
        if frame_size == 0 || audio.len() < frame_size * 3 {
            return 0.0;
        }

        // Track F0 over time
        let mut f0_track: Vec<(f64, f64)> = Vec::new(); // (time, f0)

        for (i, start) in (0..audio.len().saturating_sub(frame_size)).step_by(frame_size).enumerate() {
            let frame = &audio[start..start + frame_size];
            let f0 = self.estimate_f0(frame);
            if f0 > 0.0 {
                let time = i as f64 * frame_size as f64 / self.sample_rate as f64;
                f0_track.push((time, f0));
            }
        }

        if f0_track.len() < 3 {
            return 0.0;
        }

        // Compute slopes between consecutive frames
        let slopes: Vec<f64> = f0_track.windows(2)
            .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0 + 1e-10))
            .collect();

        // Return mean absolute slope
        let mean_slope: f64 = slopes.iter().map(|s| s.abs()).sum::<f64>() / slopes.len() as f64;
        mean_slope
    }

    /// Compute AM depth (amplitude modulation depth 0-1)
    fn compute_am_depth(&self, audio: &[f64]) -> f64 {
        let envelope = self.compute_envelope(audio);

        if envelope.is_empty() {
            return 0.0;
        }

        let max_env = envelope.iter().fold(0.0_f64, |a, &b| a.max(b));
        let min_env = envelope.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if max_env == 0.0 {
            return 0.0;
        }

        // AM depth = (max - min) / (max + min)
        (max_env - min_env) / (max_env + min_env + 1e-10)
    }

    // ========================================================================
    // NON-LINEAR FACTORS (2 features) - NEW for 45D
    // ========================================================================

    /// Compute subharmonic ratio (energy in subharmonics vs fundamental)
    fn compute_subharmonic_ratio(&self, audio: &[f64]) -> f64 {
        if audio.len() < 256 {
            return 0.0;
        }

        let f0 = self.estimate_f0(audio);
        if f0 < 50.0 {
            return 0.0;
        }

        // Compute autocorrelation
        let period = (self.sample_rate as f64 / f0) as usize;
        let half_period = period / 2;

        if half_period == 0 || period >= audio.len() {
            return 0.0;
        }

        // Correlation at F0 period (fundamental strength)
        let corr_f0: f64 = audio.iter()
            .take(audio.len() - period)
            .zip(audio.iter().skip(period))
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs();

        // Correlation at half period (subharmonic strength)
        let corr_sub: f64 = audio.iter()
            .take(audio.len() - half_period)
            .zip(audio.iter().skip(half_period))
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs();

        if corr_f0 > 0.0 {
            (corr_sub / corr_f0).min(1.0)
        } else {
            0.0
        }
    }

    /// Compute spectral entropy (measure of chaos/noise)
    fn compute_spectral_entropy(&self, spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() {
            return 0.0;
        }

        // Normalize spectrum to probability distribution
        let total: f64 = spectrum.iter().sum();
        if total == 0.0 {
            return 0.0;
        }

        let probabilities: Vec<f64> = spectrum.iter().map(|x| x / total).collect();

        // Compute Shannon entropy
        let entropy: f64 = probabilities.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum();

        // Normalize by maximum entropy (uniform distribution)
        let max_entropy = (spectrum.len() as f64).ln();

        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }
}

// ============================================================================
// PYTHON BINDINGS (PyO3)
// ============================================================================

#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pyclass(name = "ZooVoxFeatureExtractor")]
pub struct PyZooVoxFeatureExtractor {
    inner: ZooVoxFeatureExtractor,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyZooVoxFeatureExtractor {
    #[new]
    #[args(sample_rate = 44100)]
    /// Create a new ZooVoxFeatureExtractor
    ///
    /// Args:
    ///     sample_rate: Audio sample rate in Hz (default: 44100)
    fn new(sample_rate: u32) -> Self {
        Self {
            inner: ZooVoxFeatureExtractor::new(sample_rate),
        }
    }

    /// Extract 30D features from audio buffer
    ///
    /// Args:
    ///     audio: Numpy array of audio samples (f64)
    ///
    /// Returns:
    ///     Numpy array of 30 feature values
    fn extract_30d<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let audio_slice = audio.as_slice()?;
        let features = self.inner.extract(audio_slice)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Feature extraction failed: {}", e)))?;

        let vector = features.to_vector();
        Ok(PyArray1::from_vec(py, vector.to_vec()).into_py(py))
    }

    /// Extract 45D features from audio buffer (30D base + 15D new)
    ///
    /// New 15D features include:
    /// - Resonance Factors (6): Formants 1-3, Bandwidths 1-2, Dispersion
    /// - Spectral Shape Factors (4): Centroid, Spread, Skewness, Kurtosis
    /// - Modulation Factors (3): Tilt, FM Slope, AM Depth
    /// - Non-Linear Factors (2): Subharmonic Ratio, Spectral Entropy
    ///
    /// Args:
    ///     audio: Numpy array of audio samples (f64)
    ///
    /// Returns:
    ///     Numpy array of 45 feature values
    fn extract_45d<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let audio_slice = audio.as_slice()?;
        let features = self.inner.extract_45d(audio_slice)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Feature extraction failed: {}", e)))?;

        let vector = features.to_vector();
        Ok(PyArray1::from_vec(py, vector.to_vec()).into_py(py))
    }

    /// Get the configured sample rate
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Get feature names for 45D features
    #[staticmethod]
    fn feature_names_45d() -> Vec<String> {
        vec![
            // Base 30D features
            "mean_f0_hz".to_string(), "duration_ms".to_string(), "f0_range_hz".to_string(),
            "harmonic_to_noise_ratio".to_string(), "spectral_flatness".to_string(), "harmonicity".to_string(),
            "attack_time_ms".to_string(), "decay_time_ms".to_string(), "sustain_level".to_string(),
            "vibrato_rate_hz".to_string(), "vibrato_depth".to_string(), "jitter".to_string(), "shimmer".to_string(),
            "mfcc_1".to_string(), "mfcc_2".to_string(), "mfcc_3".to_string(), "mfcc_4".to_string(),
            "mfcc_5".to_string(), "mfcc_6".to_string(), "mfcc_7".to_string(), "mfcc_8".to_string(),
            "mfcc_9".to_string(), "mfcc_10".to_string(), "mfcc_11".to_string(), "mfcc_12".to_string(),
            "mfcc_13".to_string(), "spectral_flux".to_string(), "median_ici_ms".to_string(),
            "onset_rate_hz".to_string(), "ici_cv".to_string(),
            // New 15D features
            "formant_1_hz".to_string(), "formant_2_hz".to_string(), "formant_3_hz".to_string(),
            "formant_1_bandwidth".to_string(), "formant_2_bandwidth".to_string(), "formant_dispersion".to_string(),
            "spectral_centroid".to_string(), "spectral_spread".to_string(), "spectral_skewness".to_string(),
            "spectral_kurtosis".to_string(), "spectral_tilt".to_string(), "fm_slope_hz_per_sec".to_string(),
            "am_depth".to_string(), "subharmonic_ratio".to_string(), "spectral_entropy".to_string(),
        ]
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = ZooVoxFeatureExtractor::new(48000);
        assert_eq!(extractor.sample_rate(), 48000);
    }

    #[test]
    fn test_extract_empty_audio() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);
        let result = extractor.extract(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_sine_wave() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate 440 Hz sine wave
        let sample_rate = 48000.0;
        let frequency = 440.0;
        let duration = 0.5; // 500ms
        let n_samples = (sample_rate * duration) as usize;

        let audio: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate).sin() * 0.5)
            .collect();

        let features = extractor.extract(&audio).unwrap();

        // Check duration
        assert!((features.duration_ms - 500.0).abs() < 10.0);

        // Check F0 is approximately correct (within 20%)
        let f0_ratio = features.mean_f0_hz / frequency;
        assert!(f0_ratio > 0.8 && f0_ratio < 1.2, "F0 estimate: {} Hz, expected: {} Hz", features.mean_f0_hz, frequency);
    }

    #[test]
    fn test_features_to_vector() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        let audio: Vec<f64> = (0..48000)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin() * 0.5)
            .collect();

        let features = extractor.extract(&audio).unwrap();
        let vec = features.to_vector();

        assert_eq!(vec.len(), 30);
        assert!(vec[0] > 0.0); // mean_f0_hz
        assert!(vec[1] > 0.0); // duration_ms
    }

    // ========================================================================
    // 45D FEATURE EXTRACTION TESTS (TDD)
    // ========================================================================

    #[test]
    fn test_extract_45d_empty_audio() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);
        let result = extractor.extract_45d(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_45d_sine_wave() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate 440 Hz sine wave
        let sample_rate = 48000.0;
        let frequency = 440.0;
        let duration = 0.5; // 500ms
        let n_samples = (sample_rate * duration) as usize;

        let audio: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate).sin() * 0.5)
            .collect();

        let features = extractor.extract_45d(&audio).unwrap();

        // Check duration
        assert!((features.duration_ms - 500.0).abs() < 10.0);

        // Check vector length
        let vec = features.to_vector();
        assert_eq!(vec.len(), 45, "45D vector should have 45 elements");

        // Check F0 is approximately correct (within 20%)
        let f0_ratio = features.mean_f0_hz / frequency;
        assert!(f0_ratio > 0.8 && f0_ratio < 1.2, "F0 estimate: {} Hz, expected: {} Hz", features.mean_f0_hz, frequency);
    }

    #[test]
    fn test_extract_45d_fm_sweep() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate FM sweep (dolphin-like)
        let sample_rate = 48000.0;
        let duration = 0.2; // 200ms
        let n_samples = (sample_rate * duration) as usize;
        let start_freq = 5000.0;
        let end_freq = 15000.0;

        let audio: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let freq = start_freq + (end_freq - start_freq) * (i as f64 / n_samples as f64);
                (2.0 * PI * freq * t).sin() * 0.5
            })
            .collect();

        let features = extractor.extract_45d(&audio).unwrap();

        // FM slope should be positive for upward sweep
        assert!(features.fm_slope_hz_per_sec > 0.0, "FM slope should be positive for upward sweep");

        // Spectral centroid should be higher than start frequency
        assert!(features.spectral_centroid > start_freq * 0.5, "Spectral centroid should reflect sweep range");
    }

    #[test]
    fn test_extract_45d_formants() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate vowel-like sound with harmonics
        let sample_rate = 48000.0;
        let duration = 0.3;
        let n_samples = (sample_rate * duration) as usize;
        let f0 = 200.0;

        let audio: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                // Fundamental + harmonics to simulate formant structure
                let s = (2.0 * PI * f0 * t).sin()
                    + 0.5 * (2.0 * PI * f0 * 2.0 * t).sin()
                    + 0.3 * (2.0 * PI * f0 * 3.0 * t).sin()
                    + 0.2 * (2.0 * PI * f0 * 4.0 * t).sin();
                s * 0.2
            })
            .collect();

        let features = extractor.extract_45d(&audio).unwrap();

        // Formant dispersion should be reasonable for harmonic series
        // (harmonic spacing = f0)
        assert!(features.formant_dispersion >= 0.0, "Formant dispersion should be non-negative");
    }

    #[test]
    fn test_extract_45d_spectral_shape() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate broadband noise burst
        let sample_rate = 48000.0;
        let duration = 0.1;
        let n_samples = (sample_rate * duration) as usize;

        // Use deterministic "noise" based on index
        let audio: Vec<f64> = (0..n_samples)
            .map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5)
            .collect();

        let features = extractor.extract_45d(&audio).unwrap();

        // Spectral centroid should be positive
        assert!(features.spectral_centroid >= 0.0, "Spectral centroid should be non-negative");

        // Spectral spread should be positive
        assert!(features.spectral_spread >= 0.0, "Spectral spread should be non-negative");

        // Spectral entropy should be high for noise
        assert!(features.spectral_entropy > 0.5, "Spectral entropy should be high for broadband noise");
    }

    #[test]
    fn test_extract_45d_am_modulated() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        // Generate AM-modulated tone (tremolo)
        let sample_rate = 48000.0;
        let carrier_freq = 1000.0;
        let mod_rate = 5.0; // 5 Hz tremolo
        let duration = 0.5;
        let n_samples = (sample_rate * duration) as usize;

        let audio: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let am = 0.5 + 0.5 * (2.0 * PI * mod_rate * t).cos(); // AM envelope
                am * (2.0 * PI * carrier_freq * t).sin() * 0.5
            })
            .collect();

        let features = extractor.extract_45d(&audio).unwrap();

        // AM depth should be significant for tremolo
        assert!(features.am_depth > 0.3, "AM depth should be significant for modulated tone");
    }

    #[test]
    fn test_extract_45d_vector_size() {
        let mut extractor = ZooVoxFeatureExtractor::new(48000);

        let audio: Vec<f64> = (0..24000)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin() * 0.5)
            .collect();

        let features = extractor.extract_45d(&audio).unwrap();
        let vec = features.to_vector();

        assert_eq!(vec.len(), 45, "45D vector must have exactly 45 elements");
    }
}
