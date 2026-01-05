use std::time::Duration;

use crate::safety_system::SafetySystem;
use crate::watchdog_timer::WatchdogTimer;

pub struct AudioProcessor {
    sample_rate: u32,
    buffer_size: usize,
    safety_system: SafetySystem,
    watchdog: WatchdogTimer,
}

impl AudioProcessor {
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            sample_rate,
            buffer_size,
            safety_system: SafetySystem::new(90.0),
            watchdog: WatchdogTimer::new(Duration::from_millis(100)),
        }
    }

    pub fn process_buffer(&mut self, input: &[f32]) -> Vec<f32> {
        // Update watchdog
        self.watchdog.update();

        // Check safety limits
        let safety_result = self.safety_system.check_spl(input);

        if safety_result.should_mute {
            // Return zeroed buffer
            let mut output = vec![0.0; input.len()];
            self.safety_system.apply_mute(&mut output);
            return output;
        }

        // Process the audio
        let processed = self.compute_stft(input);

        processed
    }

    pub fn compute_stft(&self, signal: &[f32]) -> Vec<f32> {
        // Simple FFT implementation for baseline
        // In production, this would use a optimized FFT library

        let n = signal.len();
        use rustfft::num_complex::Complex;

        let mut complex_signal: Vec<Complex<f32>> = signal
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        fft.process(&mut complex_signal);

        // Convert magnitude to normalized features (0.0 to 1.0)
        let mut features = Vec::with_capacity(n / 2);
        for i in 0..n / 2 {
            let magnitude = (complex_signal[i].re.powi(2) + complex_signal[i].im.powi(2)).sqrt();
            // Normalize to 0.0-1.0 range
            let normalized = magnitude / n as f32;
            features.push(normalized.min(1.0));
        }

        features
    }

    // pub fn start_loopback_stream(...) - Removed to avoid complex audio stream setup
    // This would require more complex device and stream management
}