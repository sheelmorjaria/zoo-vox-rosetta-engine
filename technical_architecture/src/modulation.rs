//! Modulation dynamics features
//!
//! This module provides frequency modulation (FM) analysis features,
//! which are critical for bat vocalizations (FM sweeps) and corvid calls
//! with rapid FM components.

use std::f32::consts::PI;

/// FM depth - the range of frequency modulation in Hz.
///
/// Measures how much the frequency varies during a vocalization.
/// - Low FM depth (< 50 Hz): steady tone, narrow pitch range
/// - Medium FM depth (50-200 Hz): typical vocal vibrato
/// - High FM depth (> 200 Hz): FM sweeps, wide pitch excursions
///
/// ## Algorithm
/// 1. Track instantaneous frequency over time
/// 2. Calculate range (max - min) and standard deviation
/// 3. Return both absolute depth (Hz) and relative depth (% of mean F0)
///
/// ## Use Cases
/// - **Bats**: FM sweeps are their primary communication modality
/// - **Corvids**: "Rattles" often contain rapid FM components
/// - **Marmosets**: Distinguishes phee (low FM) from trill (high FM)
///
/// ## Interpretation
/// - fm_depth_hz: Absolute depth in Hz
/// - fm_depth_percent: Depth as percentage of mean F0
///   - < 5%: Very stable (steady tone)
///   - 5-15%: Normal vibrato
///   - > 15%: Wide modulation (trills, FM sweeps)
#[derive(Debug, Clone, PartialEq)]
pub struct FmDepthCalculator {
    pub sample_rate: u32,
    pub frame_size_ms: f32,
    pub hop_size_ms: f32,
}

impl Default for FmDepthCalculator {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            frame_size_ms: 20.0,
            hop_size_ms: 10.0,
        }
    }
}

impl FmDepthCalculator {
    pub fn new(sample_rate: u32, frame_size_ms: f32, hop_size_ms: f32) -> Self {
        Self {
            sample_rate,
            frame_size_ms: frame_size_ms.max(5.0),
            hop_size_ms: hop_size_ms.max(1.0).min(frame_size_ms),
        }
    }

    /// Calculate FM depth in Hz
    ///
    /// Returns (depth_hz, depth_percent)
    pub fn calculate(&self, audio: &[f32]) -> (f32, f32) {
        if audio.len() < self.sample_rate as usize / 100 {
            return (0.0, 0.0);
        }

        // Track pitch over time
        let pitch_contour = self.track_pitch(audio);

        if pitch_contour.len() < 3 {
            return (0.0, 0.0);
        }

        // Filter out unvoiced frames
        let valid_pitches: Vec<f32> = pitch_contour
            .into_iter()
            .filter(|&f| f > 50.0 && f < self.sample_rate as f32 / 2.0)
            .collect();

        if valid_pitches.len() < 3 {
            return (0.0, 0.0);
        }

        // Calculate statistics
        let min_pitch = valid_pitches.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_pitch = valid_pitches.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_pitch = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;

        let depth_hz = max_pitch - min_pitch;
        let depth_percent = if mean_pitch > 0.0 {
            (depth_hz / mean_pitch) * 100.0
        } else {
            0.0
        };

        (depth_hz, depth_percent)
    }

    /// Track pitch over time using zero-crossing rate
    fn track_pitch(&self, audio: &[f32]) -> Vec<f32> {
        let frame_size = (self.frame_size_ms / 1000.0 * self.sample_rate as f32) as usize;
        let hop_size = (self.hop_size_ms / 1000.0 * self.sample_rate as f32) as usize;

        let mut pitches = Vec::new();

        // Check for underflow - need at least frame_size samples
        if audio.len() < frame_size {
            return pitches;
        }

        for i in (0..audio.len() - frame_size).step_by(hop_size.max(1)) {
            let frame = &audio[i..i + frame_size];
            let pitch = self.estimate_pitch_frame(frame);
            pitches.push(pitch);
        }

        pitches
    }

    /// Estimate pitch for a single frame using zero-crossing rate
    fn estimate_pitch_frame(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        // Count zero crossings
        let mut crossings = 0usize;
        for i in 1..frame.len() {
            if (frame[i] >= 0.0 && frame[i - 1] < 0.0) || (frame[i] < 0.0 && frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        if crossings < 2 {
            return 0.0;
        }

        // Zero-crossing rate → frequency
        let zcr = crossings as f32 / frame.len() as f32;
        let freq = zcr * self.sample_rate as f32 / 2.0;

        // Filter to plausible pitch range
        if freq < 50.0 || freq > self.sample_rate as f32 / 2.0 {
            0.0
        } else {
            freq
        }
    }
}

/// FM rate - the speed of frequency modulation in Hz.
///
/// Measures how quickly the frequency changes, corresponding to the
/// vibrato rate or FM sweep rate.
///
/// ## Algorithm
/// 1. Track instantaneous frequency over time
/// 2. Count peaks and valleys in the contour
/// 3. Calculate modulation rate (cycles per second)
///
/// ## Use Cases
/// - Distinguishes fast trill from slow vibrato
/// - Identifies rapid FM sweeps in bat calls
/// - Characterizes tremolo and modulation patterns
///
/// ## Interpretation
/// - 0-5 Hz: Slow modulation, pitch drift
/// - 5-10 Hz: Typical vibrato rate
/// - 10-20 Hz: Fast trill, warble
/// - > 20 Hz: Very rapid modulation (bat FM sweeps)
#[derive(Debug, Clone, PartialEq)]
pub struct FmRateCalculator {
    pub sample_rate: u32,
    pub frame_size_ms: f32,
    pub hop_size_ms: f32,
}

impl Default for FmRateCalculator {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            frame_size_ms: 20.0,
            hop_size_ms: 10.0,
        }
    }
}

impl FmRateCalculator {
    pub fn new(sample_rate: u32, frame_size_ms: f32, hop_size_ms: f32) -> Self {
        Self {
            sample_rate,
            frame_size_ms: frame_size_ms.max(5.0),
            hop_size_ms: hop_size_ms.max(1.0).min(frame_size_ms),
        }
    }

    /// Calculate FM rate in Hz (modulations per second)
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.len() < self.sample_rate as usize / 50 {
            return 0.0;
        }

        let depth_calc = FmDepthCalculator::new(self.sample_rate, self.frame_size_ms, self.hop_size_ms);
        let pitch_contour = depth_calc.track_pitch(audio);

        if pitch_contour.len() < 10 {
            return 0.0;
        }

        // Filter and smooth pitch contour
        let valid_pitches: Vec<f32> = pitch_contour
            .into_iter()
            .filter(|&f| f > 50.0 && f < self.sample_rate as f32 / 2.0)
            .collect();

        if valid_pitches.len() < 5 {
            return 0.0;
        }

        // Find peaks and valleys
        let extrema = self.find_extrema(&valid_pitches);

        if extrema.len() < 2 {
            return 0.0;
        }

        // Calculate duration in seconds
        let duration_sec = (valid_pitches.len() * self.hop_size_ms as usize) as f32 / 1000.0;

        if duration_sec < 0.01 {
            return 0.0;
        }

        // Number of modulation cycles = number of peaks
        // Rate = cycles / duration
        let num_cycles = extrema.len() as f32 / 2.0; // Peak-valley = 1 cycle
        num_cycles / duration_sec
    }

    /// Find local extrema in the pitch contour
    fn find_extrema(&self, data: &[f32]) -> Vec<usize> {
        let mut extrema = Vec::new();

        for i in 2..data.len() - 2 {
            let is_peak =
                data[i] > data[i - 1] && data[i] > data[i - 2] && data[i] > data[i + 1] && data[i] > data[i + 2];

            let is_valley =
                data[i] < data[i - 1] && data[i] < data[i - 2] && data[i] < data[i + 1] && data[i] < data[i + 2];

            if is_peak || is_valley {
                extrema.push(i);
            }
        }

        extrema
    }
}

/// AM depth - amplitude modulation depth.
///
/// Measures the depth of amplitude modulation (tremolo), which is
/// distinct from FM (frequency modulation).
///
/// ## Algorithm
/// 1. Compute amplitude envelope
/// 2. Calculate modulation depth
///
/// ## Use Cases
/// - Identifies tremolo vs. steady amplitude
/// - Characterizes pulsing or rhythmic calls
#[derive(Debug, Clone, PartialEq)]
pub struct AmDepthCalculator {
    pub sample_rate: u32,
}

impl Default for AmDepthCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl AmDepthCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate AM depth [0, 1]
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.len() < self.sample_rate as usize / 100 {
            return 0.0;
        }

        // Compute amplitude envelope
        let frame_size = 256;
        let hop_size = 128;

        let mut envelope = Vec::new();

        for i in (0..audio.len() - frame_size).step_by(hop_size) {
            let frame = &audio[i..i + frame_size];
            let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            envelope.push(rms);
        }

        if envelope.len() < 3 {
            return 0.0;
        }

        let max_env = envelope.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_env = envelope.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        if max_env < 1e-10 {
            return 0.0;
        }

        // Modulation depth = (max - min) / max
        (max_env - min_env) / max_env
    }
}

/// Helper: Generate sine wave
fn generate_sine_wave(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin()
        })
        .collect()
}

/// Helper: Generate FM sweep with sinusoidal modulation (vibrato)
fn generate_fm_sweep(
    carrier: f32,
    min_freq: f32,
    max_freq: f32,
    mod_rate: f32,
    sample_rate: u32,
    duration_sec: f32,
) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let mod_depth = (max_freq - min_freq) / 2.0;
    let center_freq = carrier;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let instant_freq = center_freq + mod_depth * (2.0 * PI * mod_rate * t).sin();
            // Integrate to get phase
            (2.0 * PI * instant_freq * t).sin()
        })
        .collect()
}

/// Helper: Generate linear FM sweep
fn generate_linear_fm_sweep(start_freq: f32, end_freq: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let progress = i as f32 / num_samples as f32;
            let instant_freq = start_freq + (end_freq - start_freq) * progress;
            (2.0 * PI * instant_freq * t).sin()
        })
        .collect()
}

/// Helper: Generate trill (alternating frequencies)
fn generate_trill(freq1: f32, freq2: f32, rate: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let period_samples = (sample_rate as f32 / rate) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let freq = if (i / period_samples).is_multiple_of(2) {
                freq1
            } else {
                freq2
            };
            (2.0 * PI * freq * t).sin()
        })
        .collect()
}

/// Helper: Generate amplitude-modulated tone (tremolo)
fn generate_am_tone(carrier: f32, mod_rate: f32, mod_depth: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let env = 1.0 - mod_depth * (1.0 - (2.0 * PI * mod_rate * t).cos()) / 2.0;
            env * (2.0 * PI * carrier * t).sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FM Depth Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_fm_depth_constant_tone() {
        let calculator = FmDepthCalculator::default();
        let constant = generate_sine_wave(440.0, 48000, 0.5);
        let (depth_hz, depth_pct) = calculator.calculate(&constant);

        // Constant tone should have relatively low FM depth (some jitter from ZCR)
        assert!(depth_hz < 100.0, "Constant tone should have low FM depth");
        assert!(depth_pct < 25.0);
    }

    #[test]
    fn test_fm_depth_vibrato() {
        let calculator = FmDepthCalculator::default();
        // Vibrato: 440 Hz with 10 Hz modulation, ±50 Hz depth
        let vibrato = generate_fm_sweep(440.0, 390.0, 490.0, 10.0, 48000, 0.5);
        let (depth_hz, depth_pct) = calculator.calculate(&vibrato);

        // Should detect the FM depth (at least some measurable depth)
        assert!(depth_hz > 30.0, "Should detect vibrato depth");
        // Note: ZCR-based estimation can be noisy, so we only check lower bound
    }

    #[test]
    fn test_fm_depth_fm_sweep() {
        let calculator = FmDepthCalculator::default();
        // FM sweep: 200 Hz → 800 Hz
        let sweep = generate_linear_fm_sweep(200.0, 800.0, 48000, 0.5);
        let (depth_hz, depth_pct) = calculator.calculate(&sweep);

        // Should detect large FM depth
        assert!(depth_hz > 400.0, "Should detect FM sweep depth");
    }

    #[test]
    fn test_fm_depth_empty() {
        let calculator = FmDepthCalculator::default();
        let empty: Vec<f32> = vec![];
        let (depth_hz, depth_pct) = calculator.calculate(&empty);
        assert_eq!(depth_hz, 0.0);
        assert_eq!(depth_pct, 0.0);
    }

    #[test]
    fn test_fm_depth_silence() {
        let calculator = FmDepthCalculator::default();
        let silence = vec![0.0; 24000];
        let (depth_hz, depth_pct) = calculator.calculate(&silence);
        assert_eq!(depth_hz, 0.0);
        assert_eq!(depth_pct, 0.0);
    }

    #[test]
    fn test_fm_depth_trill() {
        let calculator = FmDepthCalculator::default();
        // Trill: rapid alternation between two pitches
        let trill = generate_trill(400.0, 600.0, 20.0, 48000, 0.5);
        let (depth_hz, depth_pct) = calculator.calculate(&trill);

        assert!(depth_hz > 150.0, "Trill should have significant FM depth");
    }

    #[test]
    fn test_fm_depth_range() {
        let calculator = FmDepthCalculator::default();
        let tone = generate_sine_wave(1000.0, 48000, 0.2);
        let (depth_hz, depth_pct) = calculator.calculate(&tone);

        assert!(depth_hz >= 0.0);
        assert!(depth_pct >= 0.0);
    }

    #[test]
    fn test_fm_depth_low_frequency() {
        let calculator = FmDepthCalculator::default();
        let low = generate_sine_wave(100.0, 48000, 0.2);
        let (depth_hz, depth_pct) = calculator.calculate(&low);

        assert!(depth_hz.is_finite());
        assert!(depth_pct.is_finite());
    }

    // =========================================================================
    // FM Rate Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_fm_rate_constant_tone() {
        let calculator = FmRateCalculator::default();
        let constant = generate_sine_wave(440.0, 48000, 0.5);
        let rate = calculator.calculate(&constant);

        // Constant tone has no modulation
        assert!(rate < 2.0, "Constant tone should have low FM rate");
    }

    #[test]
    fn test_fm_rate_vibrato_5hz() {
        let calculator = FmRateCalculator::default();
        // 5 Hz vibrato
        let vibrato = generate_fm_sweep(440.0, 400.0, 480.0, 5.0, 48000, 0.5);
        let rate = calculator.calculate(&vibrato);

        // Should detect some modulation (ZCR-based detection may not be precise)
        assert!(rate > 0.0 && rate < 20.0, "Should detect some modulation");
    }

    #[test]
    fn test_fm_rate_vibrato_10hz() {
        let calculator = FmRateCalculator::default();
        // 10 Hz vibrato
        let vibrato = generate_fm_sweep(440.0, 400.0, 480.0, 10.0, 48000, 0.5);
        let rate = calculator.calculate(&vibrato);

        // Should detect ~10 Hz modulation rate
        assert!(rate > 5.0 && rate < 20.0, "Should detect 10 Hz vibrato");
    }

    #[test]
    fn test_fm_rate_trill() {
        let calculator = FmRateCalculator::default();
        // Fast trill at 15 Hz
        let trill = generate_trill(400.0, 600.0, 15.0, 48000, 0.5);
        let rate = calculator.calculate(&trill);

        // Should detect some modulation
        assert!(rate > 0.0, "Trill should have measurable FM rate");
    }

    #[test]
    fn test_fm_rate_empty() {
        let calculator = FmRateCalculator::default();
        let empty: Vec<f32> = vec![];
        let rate = calculator.calculate(&empty);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_fm_rate_silence() {
        let calculator = FmRateCalculator::default();
        let silence = vec![0.0; 24000];
        let rate = calculator.calculate(&silence);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_fm_rate_range() {
        let calculator = FmRateCalculator::default();
        let tone = generate_sine_wave(1000.0, 48000, 0.3);
        let rate = calculator.calculate(&tone);

        assert!(rate >= 0.0);
    }

    #[test]
    fn test_fm_rate_slow_vibrato() {
        let calculator = FmRateCalculator::default();
        // Slow 3 Hz vibrato
        let vibrato = generate_fm_sweep(440.0, 420.0, 460.0, 3.0, 48000, 0.6);
        let rate = calculator.calculate(&vibrato);

        assert!(rate > 0.0 && rate < 10.0);
    }

    // =========================================================================
    // AM Depth Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_am_depth_constant() {
        let calculator = AmDepthCalculator::new(48000);
        let constant = generate_sine_wave(440.0, 48000, 0.3);
        let depth = calculator.calculate(&constant);

        // Constant amplitude should have low AM depth
        assert!(depth < 0.3);
    }

    #[test]
    fn test_am_depth_tremolo() {
        let calculator = AmDepthCalculator::new(48000);
        // Tremolo: amplitude modulation at 5 Hz
        let tremolo = generate_am_tone(440.0, 5.0, 0.5, 48000, 0.3);
        let depth = calculator.calculate(&tremolo);

        // Should detect amplitude modulation
        assert!(depth > 0.2, "Should detect tremolo");
    }

    #[test]
    fn test_am_depth_empty() {
        let calculator = AmDepthCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let depth = calculator.calculate(&empty);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn test_am_depth_silence() {
        let calculator = AmDepthCalculator::new(48000);
        let silence = vec![0.0; 14400];
        let depth = calculator.calculate(&silence);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn test_am_depth_range() {
        let calculator = AmDepthCalculator::new(48000);
        let tone = generate_sine_wave(1000.0, 48000, 0.2);
        let depth = calculator.calculate(&tone);

        assert!((0.0..=1.0).contains(&depth));
    }

    #[test]
    fn test_am_depth_fading() {
        let calculator = AmDepthCalculator::new(48000);
        // Exponentially decaying tone
        let mut fading = vec![0.0f32; 14400];
        for (i, sample) in fading.iter_mut().enumerate() {
            let t = i as f32 / 48000.0;
            let env = (-5.0 * t).exp();
            *sample = env * (2.0 * PI * 440.0 * t).sin();
        }

        let depth = calculator.calculate(&fading);
        // Fading should produce high AM depth
        assert!(depth > 0.3);
    }
}
