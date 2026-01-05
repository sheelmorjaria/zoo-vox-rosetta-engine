#[cfg(test)]
mod tests {
    use reflex_arc::*;
    use std::time::Duration;

    #[test]
    fn test_safety_system_initialization() {
        let safety = SafetySystem::new(90.0);
        assert_eq!(safety.get_current_spl(), 0.0);
    }

    #[test]
    fn test_watchdog_initialization() {
        let mut watchdog = WatchdogTimer::new(Duration::from_millis(100));
        assert!(watchdog.is_healthy());
        assert!(!watchdog.should_trigger());
    }

    #[test]
    fn test_audio_processor_creation() {
        let _processor = AudioProcessor::new(48000, 512);
        // Test that creation succeeds
        // We can't access private fields in tests
    }

    #[test]
    fn test_simple_audio_processing() {
        let processor = AudioProcessor::new(48000, 4);

        // Simple test signal
        let test_signal = vec![0.0, 1.0, 0.0, -1.0];

        // Process the signal
        let features = processor.compute_stft(&test_signal);

        // Should return features
        assert!(!features.is_empty());
        // All features should be between 0 and 1
        for &f in &features {
            assert!(f >= 0.0 && f <= 1.0);
        }
    }

    #[test]
    fn test_safety_system_protection() {
        let mut safety = SafetySystem::new(90.0);

        // Test quiet signal
        let quiet_signal = vec![0.1; 256];
        let result = safety.check_spl(&quiet_signal);
        assert!(!result.should_mute);

        // Test loud signal
        let loud_signal = vec![1.0; 256];
        let result2 = safety.check_spl(&loud_signal);
        assert!(result2.should_mute);
    }

    #[test]
    fn test_watchdog_functionality() {
        let mut watchdog = WatchdogTimer::new(Duration::from_millis(50));

        // Should be healthy initially
        assert!(watchdog.is_healthy());

        // Update should keep it healthy
        watchdog.update();
        assert!(watchdog.is_healthy());

        // Wait past timeout
        std::thread::sleep(Duration::from_millis(60));
        assert!(!watchdog.is_healthy());
        assert!(watchdog.should_trigger());
    }

    #[test]
    fn test_full_audio_pipeline() {
        let mut processor = AudioProcessor::new(48000, 512);

        // Generate test tone
        let test_signal: Vec<f32> = (0..512)
            .map(|i| (i as f32 / 512.0 * 2.0 * std::f32::consts::PI * 440.0).sin())
            .collect();

        // Test process_buffer method which includes watchdog and safety system
        let _processed = processor.process_buffer(&test_signal);

        // Process through audio processor
        let features = processor.compute_stft(&test_signal);
        assert!(!features.is_empty());
        assert_eq!(features.len(), 256); // Should be half the input size
    }

    #[test]
    fn test_spl_calculation() {
        // Test SPL conversion
        let mut safety = SafetySystem::new(90.0);

        // Test with 0 signal
        let spl = safety.get_current_spl();
        assert_eq!(spl, 0.0);

        // Test with non-zero signal
        let test_signal = vec![0.5; 100];
        let _result = safety.check_spl(&test_signal);

        // SPL should be positive and reasonable
        let current_spl = safety.get_current_spl();
        assert!(current_spl >= 0.0);
        assert!(current_spl < 120.0); // Reasonable SPL range
    }

    #[test]
    fn test_mute_functionality() {
        let safety = SafetySystem::new(90.0);
        let mut test_signal = vec![0.5; 100];

        safety.apply_mute(&mut test_signal);

        assert!(test_signal.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_audio_buffer_processing() {
        let mut processor = AudioProcessor::new(48000, 256);

        // Create a buffer with valid audio data
        let mut test_buffer = vec![0.0; 256];
        for i in 0..256 {
            test_buffer[i] = (i as f32 * 0.1).sin();
        }

        // Process the buffer
        let processed = processor.process_buffer(&test_buffer);

        // Should return processed buffer
        assert_eq!(processed.len(), test_buffer.len());

        // Since SPL is within limits, output shouldn't be zero
        let has_non_zero = processed.iter().any(|&x| x != 0.0);
        assert!(has_non_zero);
    }
}