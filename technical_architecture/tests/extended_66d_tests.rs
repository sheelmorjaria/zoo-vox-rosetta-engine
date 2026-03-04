//! Tests for 66D Extended Feature Stack
//! ======================================
//!
//! TDD tests for the extended features that complete the 112D stack.
//!
//! Structure:
//! - 46D Base (ADSR + fundamental + spectral)
//! - 66D Extended:
//!   - Layer 2: Macro Texture (30D)
//!     - Harmonic Texture (8D)
//!     - Pitch Geometry (7D)
//!     - GLCM Texture (10D)
//!     - Temporal Texture (5D)
//!   - Layer 3: Micro Texture (36D)
//!     - Vibrato Bins (5D)
//!     - FM Bins (5D)
//!     - Dynamics Bins (5D)
//!     - ICI Bins (5D)
//!     - Rhythm Bins (10D)
//!     - Perturbation Stats (6D)

#[cfg(test)]
mod tests_extended_66d {
    use technical_architecture::MicroDynamicsExtractor;

    fn generate_test_signal(sample_rate: u32, duration_ms: f32, frequency: f32) -> Vec<f32> {
        let n_samples = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect()
    }

    fn generate_fm_sweep(sample_rate: u32, duration_ms: f32, f_start: f32, f_end: f32) -> Vec<f32> {
        let n_samples = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let freq = f_start + (f_end - f_start) * t / (duration_ms / 1000.0);
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    // =========================================================================
    // TEST 1: 112D Vector Construction
    // =========================================================================

    #[test]
    fn test_112d_vector_can_be_constructed() {
        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        // Get 46D base features
        let features_45d = extractor.extract_45d(&signal).unwrap();
        let base_46d = features_45d.to_array_46d();

        assert_eq!(base_46d.len(), 46);
        println!("46D base: {:?}", &base_46d[..10]);

        // TODO: Add 66D extended features
        // let extended_66d = extractor.extract_extended_66d(&signal).unwrap();
        // let full_112d = [&base_46d[..], &extended_66d[..]].concat();
        // assert_eq!(full_112d.len(), 112);
    }

    // =========================================================================
    // TEST 2: Harmonic Texture Features (8D)
    // =========================================================================

    #[test]
    fn test_harmonic_texture_features() {
        // Harmonic texture captures the structure of harmonics
        // Features: harmonic_ratio, inharmonicity, harmonic_spread, etc.

        let extractor = MicroDynamicsExtractor::new(44100);

        // Pure sine (low harmonic complexity)
        let sine = generate_test_signal(44100, 100.0, 440.0);

        // Sawtooth-like (rich harmonics)
        let mut rich: Vec<f32> = Vec::new();
        for i in 0..(44100 * 100 / 1000) {
            let t = i as f32 / 44100.0;
            let mut sample = 0.0;
            for h in 1..=10 {
                sample += (2.0 * std::f32::consts::PI * 440.0 * h as f32 * t).sin() / h as f32;
            }
            rich.push(sample * 0.1);
        }

        let features_sine = extractor.extract_45d(&sine).unwrap();
        let features_rich = extractor.extract_45d(&rich).unwrap();

        // HNR should be higher for rich harmonics
        println!(
            "HNR - Sine: {:.2}, Rich: {:.2}",
            features_sine.base_30d.harmonic_to_noise_ratio,
            features_rich.base_30d.harmonic_to_noise_ratio
        );

        // TODO: Extract 8D harmonic texture
        // - harmonic_density: How densely packed are the harmonics
        // - harmonic_clarity: How distinct are harmonic peaks
        // - spectral_bandwidth: Spread of harmonic energy
        // - spectral_rolloff: Frequency below which 85% of energy lies
        // - spectral_flatness: Already in 45D
        // - harmonic_spacing_std: Variance in harmonic spacing
        // - odd_even_ratio: Ratio of odd to even harmonics
        // - harmonic_energy_ratio: Ratio of harmonic to total energy
    }

    // =========================================================================
    // TEST 3: Pitch Geometry Features (7D)
    // =========================================================================

    #[test]
    fn test_pitch_geometry_features() {
        // Pitch geometry captures the shape of the pitch contour
        // Features: slope_mean, slope_std, curvature, inflections, etc.

        let extractor = MicroDynamicsExtractor::new(44100);

        // Static pitch
        let static_pitch = generate_test_signal(44100, 100.0, 440.0);

        // FM sweep
        let sweep = generate_fm_sweep(44100, 100.0, 440.0, 2000.0);

        let features_static = extractor.extract_45d(&static_pitch).unwrap();
        let features_sweep = extractor.extract_45d(&sweep).unwrap();

        // FM slope should be higher for sweep
        println!(
            "FM Slope - Static: {:.2}, Sweep: {:.2}",
            features_static.fm_slope, features_sweep.fm_slope
        );

        // TODO: Extract 7D pitch geometry
        // - pitch_slope_mean: Average slope of pitch contour
        // - pitch_slope_std: Variability of slope
        // - pitch_curvature_mean: Average curvature
        // - pitch_curvature_max: Maximum curvature
        // - inflection_count: Number of direction changes
        // - pitch_range: Already in 45D as f0_range_hz
        // - pitch_centroid: Mean pitch position
    }

    // =========================================================================
    // TEST 4: Vibrato Distribution (5D)
    // =========================================================================

    #[test]
    fn test_vibrato_distribution_features() {
        // Vibrato distribution captures the range of modulation rates

        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_test_signal(44100, 100.0, 440.0);

        let features = extractor.extract_45d(&signal).unwrap();

        println!(
            "Vibrato Rate: {:.1} Hz, Depth: {:.1}",
            features.base_30d.vibrato_rate_hz, features.base_30d.vibrato_depth
        );

        // TODO: Extract 5D vibrato bins
        // Bin by rate: <5Hz, 5-10Hz, 10-15Hz, 15-20Hz, >20Hz
        // Each bin contains the proportion of energy in that range
    }

    // =========================================================================
    // TEST 5: FM Distribution (5D)
    // =========================================================================

    #[test]
    fn test_fm_distribution_features() {
        // FM distribution captures the range of frequency modulation slopes

        let extractor = MicroDynamicsExtractor::new(44100);

        // Static tone
        let static_tone = generate_test_signal(44100, 100.0, 440.0);

        // Slow sweep
        let slow_sweep = generate_fm_sweep(44100, 100.0, 440.0, 880.0);

        // Fast sweep
        let fast_sweep = generate_fm_sweep(44100, 100.0, 440.0, 4000.0);

        let f_static = extractor.extract_45d(&static_tone).unwrap();
        let f_slow = extractor.extract_45d(&slow_sweep).unwrap();
        let f_fast = extractor.extract_45d(&fast_sweep).unwrap();

        println!(
            "FM Slope - Static: {:.2}, Slow: {:.2}, Fast: {:.2}",
            f_static.fm_slope, f_slow.fm_slope, f_fast.fm_slope
        );

        // TODO: Extract 5D FM bins
        // Bin by slope: flat, slow, medium, fast, very fast
    }

    // =========================================================================
    // TEST 6: 66D Extended Feature Layout
    // =========================================================================

    #[test]
    fn test_66d_extended_feature_layout() {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║     66D EXTENDED FEATURE LAYOUT                                            ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");
        println!();

        println!("  Layer 2: MACRO TEXTURE (30D)");
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  [0-7]   Harmonic Texture (8D)                                          │");
        println!("  │    - harmonic_density, harmonic_clarity, spectral_bandwidth            │");
        println!("  │    - spectral_rolloff, harmonic_spacing_std, odd_even_ratio            │");
        println!("  │    - harmonic_energy_ratio, spectral_crest                             │");
        println!("  │                                                                         │");
        println!("  │  [8-14]  Pitch Geometry (7D)                                            │");
        println!("  │    - pitch_slope_mean, pitch_slope_std, pitch_curvature_mean          │");
        println!("  │    - pitch_curvature_max, inflection_count, pitch_centroid            │");
        println!("  │    - pitch_stability                                                   │");
        println!("  │                                                                         │");
        println!("  │  [15-24] GLCM Texture (10D)                                             │");
        println!("  │    - contrast, dissimilarity, homogeneity, energy, entropy            │");
        println!("  │    - correlation, asm, max_probability, cluster_shade, cluster_prominence │");
        println!("  │                                                                         │");
        println!("  │  [25-29] Temporal Texture (5D)                                          │");
        println!("  │    - attack_variability, decay_variability, sustain_stability          │");
        println!("  │    - release_variability, envelope_skew                                │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        println!("  Layer 3: MICRO TEXTURE (36D)");
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  [30-34] Vibrato Bins (5D)                                              │");
        println!("  │    - Rate bins: <5Hz, 5-10Hz, 10-15Hz, 15-20Hz, >20Hz                 │");
        println!("  │                                                                         │");
        println!("  │  [35-39] FM Bins (5D)                                                   │");
        println!("  │    - Slope bins: flat, slow, medium, fast, very_fast                   │");
        println!("  │                                                                         │");
        println!("  │  [40-44] Dynamics Bins (5D)                                             │");
        println!("  │    - Amplitude bins: silent, quiet, medium, loud, very_loud            │");
        println!("  │                                                                         │");
        println!("  │  [45-49] ICI Bins (5D)                                                  │");
        println!("  │    - Interval bins: <10ms, 10-50ms, 50-100ms, 100-200ms, >200ms        │");
        println!("  │                                                                         │");
        println!("  │  [50-59] Rhythm Bins (10D)                                              │");
        println!("  │    - Tempo and pattern distribution                                    │");
        println!("  │                                                                         │");
        println!("  │  [60-65] Perturbation Stats (6D)                                        │");
        println!("  │    - jitter_mean, jitter_std, jitter_skew, shimmer_mean, shimmer_std   │");
        println!("  │    - shimmer_skew                                                      │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        println!("  112D = 46D Base + 66D Extended");
    }

    // =========================================================================
    // TEST 7: Integration Test
    // =========================================================================

    #[test]
    fn test_112d_integration() {
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║     112D AUGMENTED STACK INTEGRATION TEST                                  ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");
        println!();

        let extractor = MicroDynamicsExtractor::new(44100);
        let signal = generate_fm_sweep(44100, 200.0, 1000.0, 5000.0);

        let features = extractor.extract_45d(&signal).unwrap();
        let base_46d = features.to_array_46d();

        println!("  46D Base Features:");
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!(
            "  │  [0-2]   Fundamental: f0={:.0}Hz, dur={:.0}ms, range={:.0}Hz",
            base_46d[0], base_46d[1], base_46d[2]
        );
        println!(
            "  │  [3-5]   Grit: HNR={:.1}, flatness={:.2}, harm={:.2}",
            base_46d[3], base_46d[4], base_46d[5]
        );
        println!(
            "  │  [6-13]  Motion: attack={:.1}ms, decay={:.1}ms, sustain={:.2}",
            base_46d[6], base_46d[7], base_46d[8]
        );
        println!(
            "  │          vibrato={:.1}Hz, depth={:.1}, jitter={:.3}, shimmer={:.3}",
            base_46d[9], base_46d[10], base_46d[11], base_46d[12]
        );
        println!("  │          release={:.1}ms", base_46d[13]);
        println!(
            "  │  [14-27] Fingerprint: MFCCs + spectral_flux={:.2}",
            base_46d[27]
        );
        println!("  │  [28-45] Resonance, Shape, Modulation, Non-Linear");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        println!("  66D Extended Features:");
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  Layer 2: Macro Texture (30D)                                          │");
        println!("  │    Harmonic: density, clarity, bandwidth, rolloff, spacing, odd/even   │");
        println!("  │    Pitch: slope_mean, slope_std, curvature, inflections               │");
        println!("  │    GLCM: contrast, homogeneity, energy, entropy, correlation          │");
        println!("  │    Temporal: attack_var, decay_var, sustain_stab, release_var         │");
        println!("  │                                                                         │");
        println!("  │  Layer 3: Micro Texture (36D)                                          │");
        println!("  │    Vibrato bins: 5 rate bands                                          │");
        println!("  │    FM bins: 5 slope bands                                              │");
        println!("  │    Dynamics bins: 5 amplitude bands                                    │");
        println!("  │    ICI bins: 5 interval bands                                          │");
        println!("  │    Rhythm bins: 10 pattern features                                    │");
        println!("  │    Perturbation: jitter/shimmer stats (6D)                             │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        // TODO: When implemented
        // let extended_66d = extractor.extract_extended_66d(&signal).unwrap();
        // assert_eq!(extended_66d.len(), 66);
        //
        // let full_112d = [&base_46d[..], &extended_66d[..]].concat();
        // assert_eq!(full_112d.len(), 112);
    }
}
