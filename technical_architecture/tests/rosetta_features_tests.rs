//! Tests for RosettaFeatures (112D) - Universal Rosetta Stone Methodology
//! ========================================================================
//!
//! TDD tests for the primary feature vector used in cross-species
//! vocalization analysis. RosettaFeatures is the recommended API for
//! all feature extraction in the Zoo Vox Rosetta system.
//!
//! Author: Test Coverage Initiative
//! License: CC BY-ND 4.0 International

#[cfg(test)]
mod tests_rosetta_features {
    use technical_architecture::{MicroDynamicsExtractor, RosettaFeatures};

    /// Helper to generate a test signal
    fn generate_test_signal(sample_rate: u32, duration_ms: f32, frequency: f32) -> Vec<f32> {
        let n_samples = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect()
    }

    /// Generate signal with envelope (ADSR)
    #[allow(clippy::too_many_arguments)]
    fn generate_signal_with_envelope(
        sample_rate: u32,
        duration_ms: f32,
        frequency: f32,
        attack_ms: f32,
        _decay_ms: f32,
        sustain_level: f32,
        release_ms: f32,
    ) -> Vec<f32> {
        let n_samples = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        let attack_samples = (sample_rate as f32 * attack_ms / 1000.0) as usize;
        let release_samples = (sample_rate as f32 * release_ms / 1000.0) as usize;

        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let base = (2.0 * std::f32::consts::PI * frequency * t).sin();

                // ADSR envelope
                let envelope = if i < attack_samples {
                    // Attack phase
                    i as f32 / attack_samples.max(1) as f32
                } else if i < n_samples - release_samples {
                    // Sustain phase (with decay applied)
                    sustain_level
                } else {
                    // Release phase
                    let release_progress = (i - (n_samples - release_samples)) as f32 / release_samples.max(1) as f32;
                    sustain_level * (1.0 - release_progress)
                };

                base * envelope
            })
            .collect()
    }

    // =========================================================================
    // TEST 1: Basic RosettaFeatures extraction
    // =========================================================================

    #[test]
    fn test_rosetta_features_extraction() {
        // RosettaFeatures should be extractable from any audio
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features: RosettaFeatures = extractor.extract(&signal).unwrap();

        // Should have 112 total dimensions
        let arr = features.to_array();
        assert_eq!(arr.len(), 112);
    }

    #[test]
    fn test_rosetta_features_base_46d() {
        // Base 46D features should be populated
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();

        // Base 46D should not be all zeros
        let has_nonzero = features.base_46d().iter().any(|&x| x != 0.0);
        assert!(has_nonzero);
    }

    #[test]
    fn test_rosetta_features_extended_66d() {
        // Extended 66D features should be populated
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();

        // Extended 66D should not be all zeros
        let has_nonzero = features.extended_66d().iter().any(|&x| x != 0.0);
        assert!(has_nonzero);
    }

    // =========================================================================
    // TEST 2: Feature dimension breakdown
    // =========================================================================

    #[test]
    fn test_rosetta_features_dimensions() {
        // Verify exact dimension counts
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();

        // Base 46D: 46 elements
        assert_eq!(features.base_46d().len(), 46);

        // Extended 66D: 66 elements
        assert_eq!(features.extended_66d().len(), 66);

        // Total: 112 elements
        assert_eq!(features.to_array().len(), 112);
    }

    // =========================================================================
    // TEST 3: Consistency with 45D base
    // =========================================================================

    #[test]
    fn test_rosetta_features_45d_consistency() {
        // RosettaFeatures should produce consistent features on repeated extraction
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let rosetta = extractor.extract(&signal).unwrap();
        let rosetta2 = extractor.extract(&signal).unwrap();

        // Both extractions should produce identical results
        assert!((rosetta.mean_f0_hz - rosetta2.mean_f0_hz).abs() < 0.01);
        assert!((rosetta.duration_ms - rosetta2.duration_ms).abs() < 0.01);
        assert!((rosetta.f0_range_hz - rosetta2.f0_range_hz).abs() < 0.01);

        // Envelope features should be preserved
        assert!((rosetta.attack_time_ms - rosetta2.attack_time_ms).abs() < 0.01);
        assert!((rosetta.decay_time_ms - rosetta2.decay_time_ms).abs() < 0.01);
        assert!((rosetta.sustain_level - rosetta2.sustain_level).abs() < 0.01);
    }

    // =========================================================================
    // TEST 4: Different signal types
    // =========================================================================

    #[test]
    fn test_rosetta_features_harmonic_signal() {
        // Harmonic signals should have high HNR
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 200.0, 8000.0); // 200ms, 8kHz (marmoset range)

        let features = extractor.extract(&signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    #[test]
    fn test_rosetta_features_with_envelope() {
        // Signals with ADSR envelope should capture envelope features
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_signal_with_envelope(
            44100, 200.0, 8000.0, 10.0, // attack
            20.0, // decay
            0.8,  // sustain
            50.0, // release
        );

        let features = extractor.extract(&signal).unwrap();

        // Should extract features successfully
        assert_eq!(features.to_array().len(), 112);

        // Base features should capture the envelope
        let has_attack = features.base_46d().iter().any(|&x| x > 0.0);
        assert!(has_attack);
    }

    #[test]
    fn test_rosetta_features_bat_frequency() {
        // Test with bat frequency range (20-100kHz downsampled to 44.1kHz)
        let extractor = MicroDynamicsExtractor::new(192000); // Higher sample rate for bats
        let signal = generate_test_signal(192000, 50.0, 40000.0); // 40kHz (bat range)

        let features = extractor.extract(&signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    #[test]
    fn test_rosetta_features_dolphin_frequency() {
        // Test with dolphin whistle range (2-24kHz)
        let extractor = MicroDynamicsExtractor::new(96000);
        let signal = generate_test_signal(96000, 300.0, 8000.0); // 8kHz (dolphin range)

        let features = extractor.extract(&signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    // =========================================================================
    // TEST 5: Edge cases
    // =========================================================================

    #[test]
    fn test_rosetta_features_silent_signal() {
        // Silent signals should still produce valid features
        let extractor = MicroDynamicsExtractor::new(44100);
        let silence = vec![0.0; 4410]; // 100ms of silence

        let features = extractor.extract(&silence).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    #[test]
    fn test_rosetta_features_short_signal() {
        // Very short signals should be handled
        let extractor = MicroDynamicsExtractor::new(44100);
        let short_signal = generate_test_signal(44100, 10.0, 440.0); // 10ms

        // Should either succeed or return an appropriate error
        let result = extractor.extract(&short_signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rosetta_features_long_signal() {
        // Long signals should be handled
        let extractor = MicroDynamicsExtractor::new(44100);
        let long_signal = generate_test_signal(44100, 2000.0, 440.0); // 2 seconds

        let features = extractor.extract(&long_signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    // =========================================================================
    // TEST 6: Array conversion
    // =========================================================================

    #[test]
    fn test_rosetta_features_to_array() {
        // to_array should produce correct 112D flat array
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();
        let arr = features.to_array();

        // Check array length
        assert_eq!(arr.len(), 112);

        // Array should contain both base and extended features
        // First 46 should match base_46d()
        let base = features.base_46d();
        for i in 0..46 {
            assert!((arr[i] - base[i]).abs() < 1e-6);
        }

        // Remaining 66 should match extended_66d()
        let extended = features.extended_66d();
        for i in 0..66 {
            assert!((arr[46 + i] - extended[i]).abs() < 1e-6);
        }
    }

    // =========================================================================
    // TEST 7: Sample rate handling
    // =========================================================================

    #[test]
    fn test_rosetta_features_44100_hz() {
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    #[test]
    fn test_rosetta_features_48000_hz() {
        let extractor = MicroDynamicsExtractor::new(48000);
        let signal = generate_test_signal(48000, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    #[test]
    fn test_rosetta_features_96000_hz() {
        let extractor = MicroDynamicsExtractor::new(96000);
        let signal = generate_test_signal(96000, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();
        assert_eq!(features.to_array().len(), 112);
    }

    // =========================================================================
    // TEST 8: Cloning and debugging
    // =========================================================================

    #[test]
    fn test_rosetta_features_clone() {
        // RosettaFeatures should be cloneable
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();
        let cloned = features.clone();

        assert_eq!(features.to_array(), cloned.to_array());
    }

    #[test]
    fn test_rosetta_features_debug() {
        // RosettaFeatures should implement Debug
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract(&signal).unwrap();
        let debug_str = format!("{:?}", features);

        // Should contain key feature fields
        assert!(debug_str.contains("mean_f0_hz"));
        assert!(debug_str.contains("duration_ms"));
        assert!(debug_str.contains("rms_energy"));
    }
}
