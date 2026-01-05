use std::f32;

pub struct SafetySystem {
    spl_threshold: f32, // dB SPL threshold
    current_spl: f32,
}

impl SafetySystem {
    pub fn new(spl_threshold: f32) -> Self {
        Self {
            spl_threshold,
            current_spl: 0.0,
        }
    }

    pub fn check_spl(&mut self, audio_buffer: &[f32]) -> SafetyCheck {
        // Calculate RMS
        let rms = self.calculate_rms(audio_buffer);

        // Convert to approximate dB SPL
        // This is a simplified conversion - in practice you'd need calibration
        let spl = self.rms_to_spl(rms);

        self.current_spl = spl;

        SafetyCheck {
            spl,
            should_mute: spl > self.spl_threshold,
            is_over_threshold: spl > self.spl_threshold,
        }
    }

    pub fn apply_mute(&self, audio_buffer: &mut [f32]) {
        // Zero out the buffer to mute audio
        for sample in audio_buffer.iter_mut() {
            *sample = 0.0;
        }
    }

    pub fn get_current_spl(&self) -> f32 {
        self.current_spl
    }

    fn calculate_rms(&self, audio_buffer: &[f32]) -> f32 {
        if audio_buffer.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = audio_buffer.iter()
            .map(|&x| x * x)
            .sum();

        let mean_square = sum_squares / audio_buffer.len() as f32;
        mean_square.sqrt()
    }

    fn rms_to_spl(&self, rms: f32) -> f32 {
        // Simplified conversion
        // Full conversion would require calibration with the specific microphone
        // and preamplifier chain

        if rms < 1e-10 {
            return 0.0;
        }

        // Approximate dB SPL calculation
        // Reference: 1 Pa = 94 dB SPL for a 1 kHz sine wave
        // This is very approximate and needs calibration
        let reference_pressure = 2e-5; // 20 μPa (reference sound pressure)
        let rms_pressure = rms * reference_pressure;

        20.0 * rms_pressure.log10() + 94.0
    }
}

pub struct SafetyCheck {
    pub spl: f32,
    pub should_mute: bool,
    pub is_over_threshold: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_system_initialization() {
        let safety = SafetySystem::new(90.0);
        assert_eq!(safety.get_current_spl(), 0.0);
    }

    #[test]
    fn test_rms_calculation() {
        let safety = SafetySystem::new(90.0);

        // Test with known signal: sin wave
        let signal = vec![0.0, 1.0, 0.0, -1.0, 0.0]; // sin wave samples
        let rms = safety.calculate_rms(&signal);

        // RMS of sin wave should be 1/sqrt(2)
        let expected_rms = 1.0 / 2.0f32.sqrt();
        assert!((rms - expected_rms).abs() < 0.01);
    }

    #[test]
    fn test_spl_check_muting() {
        let mut safety = SafetySystem::new(90.0);

        // Test quiet signal (should not mute)
        let quiet_signal = vec![0.1; 256];
        let result = safety.check_spl(&quiet_signal);
        assert!(!result.should_mute);

        // Test loud signal (should mute)
        let loud_signal = vec![1.0; 256];
        let result2 = safety.check_spl(&loud_signal);
        assert!(result2.should_mute);
    }

    #[test]
    fn test_mute_function() {
        let safety = SafetySystem::new(90.0);
        let mut test_signal = vec![0.5; 100];

        safety.apply_mute(&mut test_signal);

        assert!(test_signal.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_spl_conversion() {
        let safety = SafetySystem::new(90.0);

        // Test with 0 signal
        let spl = safety.rms_to_spl(0.0);
        assert_eq!(spl, 0.0);

        // Test with non-zero signal
        let rms = 0.5;
        let spl = safety.rms_to_spl(rms);

        // SPL should be positive and reasonable
        assert!(spl > 0.0);
        assert!(spl < 120.0); // Reasonable SPL range
    }
}