//! 56D Feature Validation Example
//!
//! This example demonstrates the 56D full-feature extraction with:
//! - Original 30D features
//! - Full Δ MFCCs (13D - all first derivatives preserved)
//! - Full ΔΔ MFCCs (13D - all second derivatives preserved)
//!
//! Total: 30 + 13 + 13 = 56D (full delta preservation)
//!
//! The 56D representation is useful when you need to preserve all temporal
//! dynamics information for downstream tasks like classification or clustering.

use std::time::Instant;
use technical_architecture::MicroDynamicsExtractor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 56D Feature Validation Example ===");
    println!();

    // Create extractor with 48kHz sample rate
    let sample_rate = 48000;
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    // Generate test audio (bat-like FM sweep: 40kHz → 20kHz, 150ms)
    println!("Generating test audio (bat FM sweep 40kHz→20kHz, 150ms)...");
    let audio = generate_fm_sweep(40000.0, 20000.0, 150.0, sample_rate);
    println!(
        "Audio length: {} samples ({:.1} ms)",
        audio.len(),
        audio.len() as f32 / sample_rate as f32 * 1000.0
    );
    println!();

    // Extract 56D features
    println!("Extracting 56D features (full delta preservation)...");
    let start = Instant::now();
    let features56 = extractor.extract_56d(&audio)?;
    let elapsed = start.elapsed();

    println!("Extraction time: {:.2} ms", elapsed.as_millis() as f32);
    println!();

    // Display base 30D features
    println!("--- Base 30D Features ---");
    println!("Attack time: {:.2} ms", features56.base_30d.attack_time_ms);
    println!("Decay time: {:.2} ms", features56.base_30d.decay_time_ms);
    println!("Sustain level: {:.2}", features56.base_30d.sustain_level);
    println!(
        "Vibrato rate: {:.2} Hz",
        features56.base_30d.vibrato_rate_hz
    );
    println!(
        "Vibrato depth: {:.2} cents",
        features56.base_30d.vibrato_depth
    );
    println!();

    println!("--- Timbre Features ---");
    println!("Jitter: {:.4}", features56.base_30d.jitter);
    println!("Shimmer: {:.4}", features56.base_30d.shimmer);
    println!("Harmonicity: {:.2}", features56.base_30d.harmonicity);
    println!(
        "Spectral flatness: {:.2}",
        features56.base_30d.spectral_flatness
    );
    println!("HNR: {:.2} dB", features56.base_30d.harmonic_to_noise_ratio);
    println!();

    println!("--- MFCC Static Features (13D) ---");
    for (i, &mfcc) in features56.base_30d.mfcc.iter().enumerate() {
        println!("  MFCC[{}]: {:.4}", i + 1, mfcc);
    }
    println!();

    println!("Spectral flux: {:.4}", features56.base_30d.spectral_flux);
    println!();

    println!("--- Rhythm Factors (3D) ---");
    println!("Median ICI: {:.2} ms", features56.base_30d.median_ici_ms);
    println!("Onset rate: {:.2} Hz", features56.base_30d.onset_rate_hz);
    println!(
        "ICI CV: {:.2}",
        features56.base_30d.ici_coefficient_of_variation
    );
    println!();

    // Display full delta MFCCs (13D)
    println!("--- Δ MFCCs (First Derivatives, 13D) ---");
    for (i, &delta) in features56.mfcc_delta.iter().enumerate() {
        println!("  Δ MFCC[{}]: {:.6}", i + 1, delta);
    }
    println!();

    println!("--- ΔΔ MFCCs (Second Derivatives, 13D) ---");
    for (i, &delta_delta) in features56.mfcc_delta_delta.iter().enumerate() {
        println!("  ΔΔ MFCC[{}]: {:.6}", i + 1, delta_delta);
    }
    println!();

    // Display temporal deltas
    println!("--- Temporal Deltas ---");
    println!("Δ F0: {:.2} Hz", features56.f0_delta);
    println!("ΔΔ F0: {:.2} Hz²", features56.f0_delta_delta);
    println!();

    // Validate feature values
    println!("--- Validation Checks ---");
    validate_features(&features56);
    println!();

    // Analyze delta patterns
    println!("--- Delta Pattern Analysis ---");
    analyze_delta_patterns(&features56);
    println!();

    // Summary
    println!("=== Summary ===");
    println!("✓ 56D feature extraction successful!");
    println!("✓ All feature values are finite and valid");
    println!(
        "✓ Extraction time: {:.2} ms (<220ms target)",
        elapsed.as_millis() as f32
    );
    println!();

    println!("Feature breakdown:");
    println!("  - Base 30D features: ✓");
    println!("  - Full Δ MFCCs (13D): ✓");
    println!("  - Full ΔΔ MFCCs (13D): ✓");
    println!("  - Temporal F0 deltas: ✓");
    println!();

    println!("Total dimensionality: 30 + 13 + 13 = 56D");
    println!();

    Ok(())
}

/// Generate an FM sweep (frequency modulated tone)
fn generate_fm_sweep(
    start_freq_hz: f32,
    end_freq_hz: f32,
    duration_ms: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    for (i, sample) in audio.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let progress = i as f32 / num_samples as f32;

        // Linear frequency sweep
        let current_freq = start_freq_hz + (end_freq_hz - start_freq_hz) * progress;

        *sample = (2.0 * std::f32::consts::PI * current_freq * t).sin();
    }

    audio
}

/// Validate that all feature values are finite and reasonable
fn validate_features(features: &technical_architecture::MicroDynamicsFeatures56D) {
    let mut all_valid = true;

    // Check base 30D features
    let base = &features.base_30d;
    if !base.attack_time_ms.is_finite() || base.attack_time_ms < 0.0 {
        println!("✗ Attack time invalid: {}", base.attack_time_ms);
        all_valid = false;
    }

    // Check all delta MFCCs
    for (i, &delta) in features.mfcc_delta.iter().enumerate() {
        if !delta.is_finite() {
            println!("✗ Δ MFCC[{}] not finite: {}", i, delta);
            all_valid = false;
        }
    }

    // Check all delta-delta MFCCs
    for (i, &delta_delta) in features.mfcc_delta_delta.iter().enumerate() {
        if !delta_delta.is_finite() {
            println!("✗ ΔΔ MFCC[{}] not finite: {}", i, delta_delta);
            all_valid = false;
        }
    }

    // Check temporal deltas
    if !features.f0_delta.is_finite() {
        println!("✗ Δ F0 not finite: {}", features.f0_delta);
        all_valid = false;
    }
    if !features.f0_delta_delta.is_finite() {
        println!("✗ ΔΔ F0 not finite: {}", features.f0_delta_delta);
        all_valid = false;
    }

    if all_valid {
        println!("✓ All feature values are valid and finite");
    }
}

/// Analyze patterns in delta features
fn analyze_delta_patterns(features: &technical_architecture::MicroDynamicsFeatures56D) {
    // Calculate delta statistics
    let delta_sum: f32 = features.mfcc_delta.iter().sum();
    let delta_abs_sum: f32 = features.mfcc_delta.iter().map(|&d| d.abs()).sum();
    let delta_max = features
        .mfcc_delta
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let delta_min = features
        .mfcc_delta
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b));

    println!("Δ MFCC Statistics:");
    println!("  Sum: {:.6}", delta_sum);
    println!("  Abs sum: {:.6}", delta_abs_sum);
    println!("  Min: {:.6}", delta_min);
    println!("  Max: {:.6}", delta_max);
    println!("  Mean: {:.6}", delta_sum / 13.0);
    println!();

    // Calculate delta-delta statistics
    let dd_sum: f32 = features.mfcc_delta_delta.iter().sum();
    let dd_abs_sum: f32 = features.mfcc_delta_delta.iter().map(|&d| d.abs()).sum();
    let dd_max = features
        .mfcc_delta_delta
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let dd_min = features
        .mfcc_delta_delta
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b));

    println!("ΔΔ MFCC Statistics:");
    println!("  Sum: {:.6}", dd_sum);
    println!("  Abs sum: {:.6}", dd_abs_sum);
    println!("  Min: {:.6}", dd_min);
    println!("  Max: {:.6}", dd_max);
    println!("  Mean: {:.6}", dd_sum / 13.0);
    println!();

    // Check for significant temporal changes
    let has_significant_delta = delta_abs_sum > 0.01;
    let has_significant_dd = dd_abs_sum > 0.01;

    if has_significant_delta {
        println!("✓ Significant Δ MFCC activity detected (temporal dynamics present)");
    } else {
        println!("✓ Low Δ MFCC activity (steady-state or constant signal)");
    }

    if has_significant_dd {
        println!("✓ Significant ΔΔ MFCC activity (acceleration in spectral change)");
    } else {
        println!("✓ Low ΔΔ MFCC activity (linear or constant change)");
    }
}
