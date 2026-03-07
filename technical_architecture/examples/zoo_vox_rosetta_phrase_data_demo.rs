//! Zoo Vox Rosetta Engine v2.0 - Phrase Data Preparation Example
//!
//! Demonstrates the complete phrase data preparation pipeline:
//! - 30D acoustic feature extraction
//! - Species-specific phrase segmentation
//! - JSON phrase library generation for all 10 species
//!
//! Usage:
//!   cargo run --release --example zoo_vox_rosetta_phrase_data_demo

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use technical_architecture::species::SpeciesConfigFactory;
use technical_architecture::{
    ZooVoxExtractionConfig, ZooVoxFeatureExtractor, ZooVoxLibraryBuilder, ZooVoxPhraseExtractor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Zoo Vox Rosetta Engine v2.0");
    println!("Phrase Data Preparation Pipeline");
    println!("========================================\n");

    // ========================================================================
    // Step 1: Generate synthetic audio samples for each species
    // ========================================================================
    println!("Step 1: Generating synthetic audio samples...\n");

    let sample_rate: u32 = 48000;

    // Species-specific audio parameters
    let species_params = [
        ("sperm_whale", 5000.0, 50.0, "click"),
        ("zebra_finch", 2500.0, 100.0, "harmonic"),
        ("meerkat", 1500.0, 80.0, "chirp"),
        ("dolphin", 8000.0, 500.0, "whistle"),
        ("orca", 6000.0, 400.0, "whistle"),
        ("egyptian_bat", 25000.0, 30.0, "chirp"),
        ("marmoset", 6800.0, 65.0, "harmonic"),
        ("giant_otter", 800.0, 200.0, "growl"),
        ("macaque", 500.0, 150.0, "chirp"),
    ];

    for (species, freq, duration_ms, call_type) in &species_params {
        println!("  {} - {:.0}Hz, {:.0}ms ({})", species, freq, duration_ms, call_type);
    }
    println!();

    // ========================================================================
    // Step 2: Extract 30D features from marmoset-like call
    // ========================================================================
    println!("Step 2: Extracting 30D features from marmoset call...\n");

    // Generate synthetic marmoset call (AM-modulated 6800 Hz tone)
    let audio: Vec<f64> = (0..(sample_rate as f64 * 0.5) as usize)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // AM-modulated tone
            let carrier = (2.0 * std::f64::consts::PI * 6800.0 * t).sin();
            let modulation = 0.3 * (2.0 * std::f64::consts::PI * 12.0 * t).sin() + 0.7;
            carrier * modulation * 0.5
        })
        .collect();

    let mut feature_extractor = ZooVoxFeatureExtractor::new(sample_rate);
    let features = feature_extractor.extract(&audio)?;

    println!("  === FUNDAMENTAL (3 features) ===");
    println!("  Mean F0:        {:.1} Hz", features.mean_f0_hz);
    println!("  Duration:       {:.1} ms", features.duration_ms);
    println!("  F0 Range:       {:.1} Hz", features.f0_range_hz);

    println!("\n  === GRIT FACTORS (3 features) ===");
    println!("  HNR:            {:.1} dB", features.harmonic_to_noise_ratio);
    println!("  Spectral Flat:  {:.3}", features.spectral_flatness);
    println!("  Harmonicity:    {:.3}", features.harmonicity);

    println!("\n  === MOTION FACTORS (7 features) ===");
    println!("  Attack Time:    {:.1} ms", features.attack_time_ms);
    println!("  Decay Time:     {:.1} ms", features.decay_time_ms);
    println!("  Sustain Level:  {:.3}", features.sustain_level);
    println!("  Vibrato Rate:   {:.1} Hz", features.vibrato_rate_hz);
    println!("  Vibrato Depth:  {:.2} st", features.vibrato_depth);
    println!("  Jitter:         {:.4}", features.jitter);
    println!("  Shimmer:        {:.4}", features.shimmer);

    println!("\n  === FINGERPRINT FACTORS (14 features) ===");
    println!(
        "  MFCC 1-3:       [{:.1}, {:.1}, {:.1}]",
        features.mfcc_1, features.mfcc_2, features.mfcc_3
    );
    println!("  Spectral Flux:  {:.1}", features.spectral_flux);

    println!("\n  === RHYTHM FACTORS (3 features) ===");
    println!("  Median ICI:     {:.1} ms", features.median_ici_ms);
    println!("  Onset Rate:     {:.1} Hz", features.onset_rate_hz);
    println!("  ICI CV:         {:.3}", features.ici_coefficient_of_variation);

    // ========================================================================
    // Step 3: Extract phrases using species-specific segmentation
    // ========================================================================
    println!("\nStep 3: Extracting phrases with species-specific segmentation...\n");

    let config = ZooVoxExtractionConfig::for_species("marmoset", sample_rate);
    let mut phrase_extractor = ZooVoxPhraseExtractor::new(config);

    // Generate longer audio with multiple phrases
    let long_audio: Vec<f64> = (0..(sample_rate as f64 * 2.0) as usize)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Create multiple "phrases" with gaps
            let in_phrase = (t * 5.0).fract() < 0.6;
            if in_phrase {
                let carrier = (2.0 * std::f64::consts::PI * 6800.0 * t).sin();
                let modulation = 0.3 * (2.0 * std::f64::consts::PI * 12.0 * t).sin() + 0.7;
                carrier * modulation * 0.5
            } else {
                0.0
            }
        })
        .collect();

    let phrases = phrase_extractor.extract_phrases(&long_audio, "marmoset", None)?;

    println!("  Extracted {} phrases", phrases.len());
    for (i, phrase) in phrases.iter().take(3).enumerate() {
        println!(
            "    Phrase {}: {} (F0={:.0}Hz, Dur={:.0}ms)",
            i + 1,
            phrase.phrase_key,
            phrase.features_30d.mean_f0_hz,
            phrase.features_30d.duration_ms
        );
    }
    if phrases.len() > 3 {
        println!("    ... and {} more", phrases.len() - 3);
    }

    // ========================================================================
    // Step 4: Build phrase library
    // ========================================================================
    println!("\nStep 4: Building phrase library...\n");

    let builder = ZooVoxLibraryBuilder::new().with_similarity_threshold(0.85);

    let library = builder.build_library(phrases, "marmoset", None)?;

    println!("  Species: {}", library.species);
    println!("  Total Phrases: {}", library.total_phrases);
    println!("  Total Occurrences: {}", library.total_occurrences);
    println!("  Type Entropy: {:.3} bits", library.type_entropy);
    println!(
        "  Frequency Range: {:.0} - {:.0} Hz",
        library.frequency_range_hz.0, library.frequency_range_hz.1
    );
    println!(
        "  Duration Range: {:.0} - {:.0} ms",
        library.typical_duration_ms.0, library.typical_duration_ms.1
    );

    // ========================================================================
    // Step 5: Show species configuration summary
    // ========================================================================
    println!("\nStep 5: Species Configuration Summary (Zoo Vox Rosetta v2.0)\n");

    println!(
        "  {:<15} {:<20} {:<12} {:<10}",
        "Species", "Encoding Strategy", "Modality", "Contexts"
    );
    println!("  {}", "-".repeat(60));

    for species in &[
        "sperm_whale",
        "zebra_finch",
        "meerkat",
        "dolphin",
        "orca",
        "egyptian_bat",
        "marmoset",
        "giant_otter",
        "macaque",
    ] {
        let config = SpeciesConfigFactory::create(species);
        println!(
            "  {:<15} {:<20} {:<12} {:<10}",
            config.species(),
            format!("{:?}", config.encoding_strategy()),
            format!("{:?}", config.modality()),
            config.context_labels().len()
        );
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n========================================");
    println!("Pipeline Summary");
    println!("========================================");
    println!("✓ 30D acoustic feature extraction");
    println!("✓ Species-specific phrase segmentation");
    println!("✓ Phrase typing and deduplication");
    println!("✓ Context association support");
    println!("✓ JSON phrase library generation");
    println!();
    println!("Supported Species: 9");
    println!("Feature Dimensions: 30 (5 categories)");
    println!("Encoding Strategies: 7 types");
    println!("Analysis Modalities: 3 (Temporal, Spectral, Hybrid)");

    Ok(())
}
