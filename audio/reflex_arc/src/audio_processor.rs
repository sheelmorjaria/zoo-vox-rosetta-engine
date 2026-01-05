use cpal::{
    Device, Host, InputCallbackInfo, SampleFormat, Stream, StreamConfig,
    SupportedStreamConfig,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
        let mut complex_signal: Vec<rustfft::Complex<f32>> = signal
            .iter()
            .map(|&x| rustfft::Complex { re: x, im: 0.0 })
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

    pub fn start_loopback_stream(
        &mut self,
        input_device: &Device,
        output_device: &Device,
    ) -> Result<Stream, cpal::StreamError> {
        // Get supported input config
        let input_config = input_device
            .default_input_config()
            .expect("Failed to get default input config");

        let sample_rate = self.sample_rate as u32;
        let channels = input_config.channels() as usize;

        // Match our required format
        let supported_format = input_config
            .sample_format()
            .specific_sample_format()
            .clone();

        // Build input stream config
        let input_config = StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Start the loopback stream
        let input_stream = input_device.build_input_stream(
            &input_config,
            supported_format,
            move |data: &[f32], _: &InputCallbackInfo| {
                // Process input and route to output
                // This is where the actual loopback would happen
                // For now, we'll just update the watchdog
                self.watchdog.update();
            },
            None,
        )?;

        Ok(input_stream)
    }
}