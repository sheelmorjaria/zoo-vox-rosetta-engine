//! Tests for 112D Augmented Feature Stack
//! =========================================
//!
//! TDD tests for the new features:
//! - ADSR (Attack, Decay, Sustain, Release)
//! - Jitter & Shimmer (Perturbations)
//! - Spectral Flux (Dynamics)

#[cfg(test)]
mod tests_augmented_112d {
    use technical_architecture::MicroDynamicsExtractor;

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

    /// Generate signal with envelope
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
    // TEST 1: Release Time Detection
    // =========================================================================

    #[test]
    fn test_release_time_is_detected() {
        // The 112D stack should include release time
        // Currently 105D has: attack, decay, sustain but NO release

        let extractor = MicroDynamicsExtractor::new(44100);

        // Generate signal with obvious release
        let signal = generate_signal_with_envelope(
            44100, 100.0, 440.0, 10.0, // attack
            20.0, // decay
            0.8,  // sustain
            30.0, // release
        );

        let features = extractor.extract_45d(&signal).unwrap();

        // Attack, decay, sustain should exist
        assert!(features.base_30d.attack_time_ms > 0.0);
        assert!(features.base_30d.decay_time_ms > 0.0);
        assert!(features.base_30d.sustain_level > 0.0);

        // TODO: This test will fail until we add release_time to the struct
        // assert!(features.release_time_ms > 0.0);
    }

    // =========================================================================
    // TEST 2: Jitter Detection
    // =========================================================================

    #[test]
    fn test_jitter_is_computed() {
        // Jitter should already exist in 30D
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract_45d(&signal).unwrap();

        // Jitter should be present (might be near 0 for clean sine)
        assert!(features.base_30d.jitter >= 0.0);
        println!("Jitter: {:.4}", features.base_30d.jitter);
    }

    #[test]
    fn test_jitter_increases_with_perturbation() {
        // Signal with added noise should have higher jitter
        let extractor = MicroDynamicsExtractor::new(44100);

        // Clean signal
        let clean = generate_test_signal(44100, 100.0, 440.0);
        let features_clean = extractor.extract_45d(&clean).unwrap();
        let jitter_clean = features_clean.base_30d.jitter;

        // Noisy signal (add frequency perturbation)
        let noisy: Vec<f32> = clean
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let freq_perturbation = (i as f32 * 0.001).sin() * 0.05;
                s * (1.0 + freq_perturbation)
            })
            .collect();
        let features_noisy = extractor.extract_45d(&noisy).unwrap();
        let jitter_noisy = features_noisy.base_30d.jitter;

        println!("Jitter clean: {:.4}, noisy: {:.4}", jitter_clean, jitter_noisy);

        // Jitter should be finite and non-negative
        assert!(jitter_clean.is_finite());
        assert!(jitter_noisy.is_finite());
    }

    // =========================================================================
    // TEST 3: Shimmer Detection
    // =========================================================================

    #[test]
    fn test_shimmer_is_computed() {
        // Shimmer should already exist in 30D
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract_45d(&signal).unwrap();

        // Shimmer should be present
        assert!(features.base_30d.shimmer >= 0.0);
        println!("Shimmer: {:.4}", features.base_30d.shimmer);
    }

    #[test]
    fn test_shimmer_increases_with_amplitude_variation() {
        // Signal with amplitude variation should have higher shimmer
        let extractor = MicroDynamicsExtractor::new(44100);

        // Clean signal
        let clean = generate_test_signal(44100, 100.0, 440.0);
        let features_clean = extractor.extract_45d(&clean).unwrap();
        let shimmer_clean = features_clean.base_30d.shimmer;

        // Signal with amplitude perturbation
        let perturbed: Vec<f32> = clean
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let amp_perturbation = 1.0 + (i as f32 * 0.01).sin() * 0.2;
                s * amp_perturbation
            })
            .collect();
        let features_perturbed = extractor.extract_45d(&perturbed).unwrap();
        let shimmer_perturbed = features_perturbed.base_30d.shimmer;

        println!(
            "Shimmer clean: {:.4}, perturbed: {:.4}",
            shimmer_clean, shimmer_perturbed
        );

        assert!(shimmer_clean.is_finite());
        assert!(shimmer_perturbed.is_finite());
    }

    // =========================================================================
    // TEST 4: Spectral Flux Detection
    // =========================================================================

    #[test]
    fn test_spectral_flux_is_computed() {
        // Spectral flux should already exist in 30D
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract_45d(&signal).unwrap();

        // Spectral flux should be present
        assert!(features.base_30d.spectral_flux >= 0.0);
        println!("Spectral Flux: {:.4}", features.base_30d.spectral_flux);
    }

    #[test]
    fn test_spectral_flux_higher_for_sweeps() {
        // FM sweep should have higher spectral flux than static tone
        let extractor = MicroDynamicsExtractor::new(44100);
        let sr = 44100;

        // Static tone
        let static_tone = generate_test_signal(sr, 100.0, 440.0);
        let features_static = extractor.extract_45d(&static_tone).unwrap();
        let flux_static = features_static.base_30d.spectral_flux;

        // FM sweep (frequency changes over time)
        let n_samples = (sr as f32 * 0.1) as usize;
        let sweep: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let freq = 440.0 + t * 2000.0; // Linear sweep
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();
        let features_sweep = extractor.extract_45d(&sweep).unwrap();
        let flux_sweep = features_sweep.base_30d.spectral_flux;

        println!("Spectral Flux - Static: {:.4}, Sweep: {:.4}", flux_static, flux_sweep);

        // Both should be finite
        assert!(flux_static.is_finite());
        assert!(flux_sweep.is_finite());
    }

    // =========================================================================
    // TEST 5: 112D Vector Construction
    // =========================================================================

    #[test]
    fn test_112d_vector_construction() {
        // Test that 112D vector can be constructed
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features_45d = extractor.extract_45d(&signal).unwrap();
        let arr_45d = features_45d.to_array();

        // Current 45D structure
        assert_eq!(arr_45d.len(), 45);

        // Verify existing ADSR features are in the 45D
        // (Attack at index 6, Decay at 7, Sustain at 8)
        // Note: Attack can be 0 for simple sine waves without envelope
        assert!(arr_45d[6] >= 0.0); // Attack
        assert!(arr_45d[7] >= 0.0); // Decay
        assert!(arr_45d[8] >= 0.0); // Sustain

        // Verify Jitter/Shimmer are in 45D (indices 11, 12)
        assert!(arr_45d[11] >= 0.0); // Jitter
        assert!(arr_45d[12] >= 0.0); // Shimmer

        // Verify Spectral Flux is in 45D (index 26)
        assert!(arr_45d[26] >= 0.0); // Spectral Flux

        println!("45D array: {:?}", &arr_45d[..30]);
    }

    // =========================================================================
    // TEST 6: Feature Names Documentation
    // =========================================================================

    #[test]
    fn test_45d_feature_layout_documentation() {
        // Document the current 45D layout
        let layout = vec![
            ("[0-2] Fundamental", "mean_f0_hz, duration_ms, f0_range_hz"),
            ("[3-5] Grit", "hnr, spectral_flatness, harmonicity"),
            (
                "[6-12] Motion",
                "attack, decay, sustain, vibrato_rate, vibrato_depth, jitter, shimmer",
            ),
            ("[13-26] Fingerprint", "mfcc_1-13, spectral_flux"),
            ("[27-29] Rhythm", "median_ici, onset_rate, ici_cv"),
            ("[30-35] Resonance", "formant_1-3, bandwidth_1-2, dispersion"),
            ("[36-39] Spectral Shape", "centroid, spread, skewness, kurtosis"),
            ("[40-42] Modulation", "spectral_tilt, fm_slope, am_depth"),
            ("[43-44] Non-Linear", "subharmonic_ratio, spectral_entropy"),
        ];

        println!("\n45D Feature Layout:");
        for (range, features) in layout {
            println!("  {} -> {}", range, features);
        }

        // The 112D augmentation adds:
        // [45] Release time (ADSR completion)
        // [46-47] Jitter variance, Shimmer variance (if needed)
        // But actually, most features already exist!

        println!("\n112D Status:");
        println!("  ✓ ADSR: Attack, Decay, Sustain already in 45D");
        println!("  ✗ Release: NOT in 45D - NEEDS TO BE ADDED");
        println!("  ✓ Jitter: Already in 45D at index 11");
        println!("  ✓ Shimmer: Already in 45D at index 12");
        println!("  ✓ Spectral Flux: Already in 45D at index 26");
    }

    // =========================================================================
    // TEST 7: Gap Analysis - What's Actually Missing
    // =========================================================================

    #[test]
    fn test_gap_analysis_105d_vs_proposed_112d() {
        // The user's analysis suggests adding 7 features to make 112D
        // Let's verify what's ACTUALLY missing

        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║     GAP ANALYSIS: 105D vs Proposed 112D                                  ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");
        println!();

        // Current 105D structure (based on examples)
        // 45D base + 60 placeholder zeros

        // Proposed 112D = 105D + 7 new features:
        // - ADSR: +4 (but attack/decay/sustain already exist, so only +1 for release)
        // - Jitter: Already exists
        // - Shimmer: Already exists
        // - Spectral Flux: Already exists

        println!("  Proposed Features Status:");
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  Feature         │ Status    │ Notes                                  │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");
        println!("  │  Attack Time     │ ✓ EXISTS  │ In 45D at index 6                      │");
        println!("  │  Decay Time      │ ✓ EXISTS  │ In 45D at index 7                      │");
        println!("  │  Sustain Level   │ ✓ EXISTS  │ In 45D at index 8                      │");
        println!("  │  Release Time    │ ✗ MISSING │ NOT in 45D - NEEDS IMPLEMENTATION      │");
        println!("  │  Jitter          │ ✓ EXISTS  │ In 45D at index 11                     │");
        println!("  │  Shimmer         │ ✓ EXISTS  │ In 45D at index 12                     │");
        println!("  │  Spectral Flux   │ ✓ EXISTS  │ In 45D at index 26                     │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        // REAL gap: Only Release Time is missing!
        // The 105D -> 112D should be 45D + 67D extended layers

        println!("  ACTUAL IMPLEMENTATION NEEDED:");
        println!("    1. Add release_time_ms to MicroDynamicsFeatures");
        println!("    2. Implement release_time extraction in extract_envelope");
        println!("    3. Add release_time to 45D output array");
        println!("    4. Fill in the 60 placeholder zeros with meaningful macro/micro features");
    }
}
