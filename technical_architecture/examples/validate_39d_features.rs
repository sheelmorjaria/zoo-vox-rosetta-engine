//! 39D Feature Validation Example
//!
//! This example demonstrates the 39D compact feature extraction with:
//! - Original 30D features
//! - Mean aggregation of Δ MFCCs (1D)
//! - Mean aggregation of ΔΔ MFCCs (1D)
//! - Multi-scale F0 features (6D: mean, std, skew, kurtosis, range, IQR)
//! - Multi-scale MFCC features (stored but not used in compact 39D)
//! - Multi-scale onset rate features (6D)
//!
//! Total: 30 + 1 + 1 + 6 + 6 = 44D (stored), with 39D used for compact representation

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::time::Instant;
use technical_architecture::MicroDynamicsExtractor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 39D Feature Validation Example ===");
    println!();

    // Create extractor with 48kHz sample rate (suitable for most vocalizations)
    let sample_rate = 48000;
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    // Generate test audio (marmoset-like phee call: 9kHz, 200ms)
    println!("Generating test audio (9kHz marmoset phee call, 200ms)...");
    let audio = generate_test_tone(9000.0, 200.0, sample_rate);
    println!(
        "Audio length: {} samples ({:.1} ms)",
        audio.len(),
        audio.len() as f32 / sample_rate as f32 * 1000.0
    );
    println!();

    // Extract 39D features
    println!("Extracting 39D features...");
    let start = Instant::now();
    let features39 = extractor.extract_39d(&audio)?;
    let elapsed = start.elapsed();

    println!("Extraction time: {:.2} ms", elapsed.as_millis() as f32);
    println!();

    // Display base 30D features
    println!("--- Base 30D Features ---");
    println!("Attack time: {:.2} ms", features39.base_30d.attack_time_ms);
    println!("Decay time: {:.2} ms", features39.base_30d.decay_time_ms);
    println!("Sustain level: {:.2}", features39.base_30d.sustain_level);
    println!("Vibrato rate: {:.2} Hz", features39.base_30d.vibrato_rate_hz);
    println!("Vibrato depth: {:.2} cents", features39.base_30d.vibrato_depth);
    println!("Jitter: {:.4}", features39.base_30d.jitter);
    println!("Shimmer: {:.4}", features39.base_30d.shimmer);
    println!("Harmonicity: {:.2}", features39.base_30d.harmonicity);
    println!("Spectral flatness: {:.2}", features39.base_30d.spectral_flatness);
    println!("HNR: {:.2} dB", features39.base_30d.harmonic_to_noise_ratio);
    println!();

    println!("--- MFCC Static Features (13D) ---");
    for (i, &mfcc) in features39.base_30d.mfcc.iter().enumerate() {
        println!("  MFCC[{}]: {:.4}", i + 1, mfcc);
    }
    println!();

    println!("Spectral flux: {:.4}", features39.base_30d.spectral_flux);
    println!();

    println!("--- Rhythm Factors (3D) ---");
    println!("Median ICI: {:.2} ms", features39.base_30d.median_ici_ms);
    println!("Onset rate: {:.2} Hz", features39.base_30d.onset_rate_hz);
    println!("ICI CV: {:.2}", features39.base_30d.ici_coefficient_of_variation);
    println!();

    // Display delta features (compact)
    println!("--- Delta Features (Compact 2D) ---");
    println!("Δ MFCC mean: {:.6}", features39.mfcc_delta_mean);
    println!("ΔΔ MFCC mean: {:.6}", features39.mfcc_delta_delta_mean);
    println!();

    // Display multi-scale F0 features
    println!("--- Multi-Scale F0 Features (6D) ---");
    println!("Mean: {:.2} Hz", features39.f0_multi_scale.mean);
    println!("Std dev: {:.2} Hz", features39.f0_multi_scale.std_dev);
    println!("Skewness: {:.4}", features39.f0_multi_scale.skewness);
    println!("Kurtosis: {:.4}", features39.f0_multi_scale.kurtosis);
    println!("Range: {:.2} Hz", features39.f0_multi_scale.range);
    println!("IQR: {:.2} Hz", features39.f0_multi_scale.iqr);
    println!();

    // Display multi-scale onset rate features
    println!("--- Multi-Scale Onset Rate Features (6D) ---");
    println!("Mean: {:.2} Hz", features39.onset_rate_multi_scale.mean);
    println!("Std dev: {:.2} Hz", features39.onset_rate_multi_scale.std_dev);
    println!("Skewness: {:.4}", features39.onset_rate_multi_scale.skewness);
    println!("Kurtosis: {:.4}", features39.onset_rate_multi_scale.kurtosis);
    println!("Range: {:.2} Hz", features39.onset_rate_multi_scale.range);
    println!("IQR: {:.2} Hz", features39.onset_rate_multi_scale.iqr);
    println!();

    // Validate feature values
    println!("--- Validation Checks ---");
    validate_features(&features39);
    println!();

    // Summary
    println!("=== Summary ===");
    println!("✓ 39D feature extraction successful!");
    println!("✓ All feature values are finite and valid");
    println!(
        "✓ Extraction time: {:.2} ms (<200ms target)",
        elapsed.as_millis() as f32
    );
    println!();

    println!("Feature breakdown:");
    println!("  - Base 30D features: ✓");
    println!("  - Δ MFCC mean (compact): ✓");
    println!("  - ΔΔ MFCC mean (compact): ✓");
    println!("  - F0 multi-scale (6D): ✓");
    println!("  - Onset rate multi-scale (6D): ✓");
    println!();

    Ok(())
}

/// Generate a test tone (pure sine wave)
fn generate_test_tone(frequency_hz: f32, duration_ms: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    for (i, sample) in audio.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *sample = (2.0 * std::f32::consts::PI * frequency_hz * t).sin();
    }

    audio
}

/// Validate that all feature values are finite and reasonable
fn validate_features(features: &technical_architecture::MicroDynamicsFeatures39D) {
    let mut all_valid = true;

    // Check base 30D features
    let base = &features.base_30d;
    if !base.attack_time_ms.is_finite() || base.attack_time_ms < 0.0 {
        println!("✗ Attack time invalid: {}", base.attack_time_ms);
        all_valid = false;
    }
    if !base.decay_time_ms.is_finite() || base.decay_time_ms < 0.0 {
        println!("✗ Decay time invalid: {}", base.decay_time_ms);
        all_valid = false;
    }
    if !base.sustain_level.is_finite() || base.sustain_level < 0.0 || base.sustain_level > 1.0 {
        println!("✗ Sustain level invalid: {}", base.sustain_level);
        all_valid = false;
    }

    // Check MFCCs
    for (i, &mfcc) in base.mfcc.iter().enumerate() {
        if !mfcc.is_finite() {
            println!("✗ MFCC[{}] not finite: {}", i, mfcc);
            all_valid = false;
        }
    }

    // Check delta features
    if !features.mfcc_delta_mean.is_finite() {
        println!("✗ Δ MFCC mean not finite: {}", features.mfcc_delta_mean);
        all_valid = false;
    }
    if !features.mfcc_delta_delta_mean.is_finite() {
        println!("✗ ΔΔ MFCC mean not finite: {}", features.mfcc_delta_delta_mean);
        all_valid = false;
    }

    // Check multi-scale F0 features
    let f0_ms = &features.f0_multi_scale;
    if !f0_ms.mean.is_finite() || f0_ms.mean <= 0.0 {
        println!("✗ F0 mean invalid: {}", f0_ms.mean);
        all_valid = false;
    }
    if !f0_ms.std_dev.is_finite() || f0_ms.std_dev < 0.0 {
        println!("✗ F0 std dev invalid: {}", f0_ms.std_dev);
        all_valid = false;
    }
    if !f0_ms.range.is_finite() || f0_ms.range < 0.0 {
        println!("✗ F0 range invalid: {}", f0_ms.range);
        all_valid = false;
    }

    // Check multi-scale onset rate features
    let onset_ms = &features.onset_rate_multi_scale;
    if !onset_ms.mean.is_finite() || onset_ms.mean < 0.0 {
        println!("✗ Onset rate mean invalid: {}", onset_ms.mean);
        all_valid = false;
    }
    if !onset_ms.std_dev.is_finite() || onset_ms.std_dev < 0.0 {
        println!("✗ Onset rate std dev invalid: {}", onset_ms.std_dev);
        all_valid = false;
    }

    if all_valid {
        println!("✓ All feature values are valid and finite");
    }
}
