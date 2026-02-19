//! Zoo Vox Rosetta Engine v2.0 - Within-Call Phrase Discovery Demo
//!
//! Demonstrates the acoustic similarity-based phrase discovery:
//! - Uses AcousticSimilarityEngine for pairwise similarity
//! - Discovers phrase types within single vocalizations
//! - Computes transition matrices and motifs
//! - Compares to simple key-based grouping
//!
//! Usage:
//!   cargo run --release --example zoo_vox_within_call_demo

use technical_architecture::species::SpeciesConfigFactory;
use technical_architecture::{
    AcousticFeatures30D, PhrasePrototype, SimilarityBasedLibraryBuilder, WithinCallAnalyzer,
    WithinCallConfig, ZooVoxExtractionConfig, ZooVoxFeatureExtractor, ZooVoxLibraryBuilder,
    ZooVoxPhraseExtractor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("Zoo Vox Rosetta Engine v2.0");
    println!("Within-Call Phrase Discovery Demo");
    println!("Acoustic Similarity Engine Integration");
    println!("========================================\n");

    let sample_rate: u32 = 48000;

    // ========================================================================
    // Step 1: Generate synthetic multi-phrase vocalization with gaps
    // ========================================================================
    println!("Step 1: Generating synthetic multi-phrase vocalization...\n");

    // Create a 5-second vocalization with multiple phrases separated by gaps
    // Each phrase is ~200ms with 100ms gaps
    let total_duration = 5.0;
    let phrase_duration = 0.2; // 200ms phrases
    let gap_duration = 0.1; // 100ms gaps
    let total_samples = (sample_rate as f64 * total_duration) as usize;

    let audio: Vec<f64> = (0..total_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;

            // Calculate which segment we're in (phrase + gap = 300ms cycle)
            let cycle_duration = phrase_duration + gap_duration;
            let cycle_position = t % cycle_duration;
            let cycle_index = (t / cycle_duration) as usize;

            // Only produce sound during phrase portion (first 200ms of each 300ms cycle)
            if cycle_position < phrase_duration {
                // Different frequencies for different cycles to create phrase types
                match cycle_index % 3 {
                    0 => {
                        // Phrase type A: 6800 Hz AM-modulated (phee-like)
                        let carrier = (2.0 * std::f64::consts::PI * 6800.0 * t).sin();
                        let modulation = 0.3 * (2.0 * std::f64::consts::PI * 12.0 * t).sin() + 0.7;
                        carrier * modulation * 0.5
                    }
                    1 => {
                        // Phrase type B: 8500 Hz with faster modulation (twitter-like)
                        let carrier = (2.0 * std::f64::consts::PI * 8500.0 * t).sin();
                        let modulation = 0.5 * (2.0 * std::f64::consts::PI * 30.0 * t).sin() + 0.5;
                        carrier * modulation * 0.5
                    }
                    _ => {
                        // Phrase type C: 5200 Hz with low modulation (tsik-like)
                        let carrier = (2.0 * std::f64::consts::PI * 5200.0 * t).sin();
                        let modulation = 0.1 * (2.0 * std::f64::consts::PI * 8.0 * t).sin() + 0.9;
                        carrier * modulation * 0.5
                    }
                }
            } else {
                // Gap: silence
                0.0
            }
        })
        .collect();

    let num_phrases = (total_duration / (phrase_duration + gap_duration)) as usize;
    println!(
        "  Generated {:.0}-second synthetic vocalization",
        total_duration
    );
    println!(
        "  Phrase duration: {}ms, Gap duration: {}ms",
        phrase_duration * 1000.0,
        gap_duration * 1000.0
    );
    println!("  Expected ~{} phrases:", num_phrases);
    println!("    - Type A (6800 Hz, phee-like)");
    println!("    - Type B (8500 Hz, twitter-like)");
    println!("    - Type C (5200 Hz, tsik-like)\n");

    // ========================================================================
    // Step 2: Extract phrases using standard pipeline
    // ========================================================================
    println!("Step 2: Extracting phrases with standard segmentation...\n");

    let config = ZooVoxExtractionConfig::for_species("marmoset", sample_rate);
    let mut phrase_extractor = ZooVoxPhraseExtractor::new(config);

    let phrases = phrase_extractor.extract_phrases(&audio, "marmoset", None)?;

    println!("  Extracted {} phrase candidates", phrases.len());
    for (i, phrase) in phrases.iter().take(5).enumerate() {
        println!(
            "    {}: F0={:.0}Hz, Dur={:.0}ms, Key={}",
            i + 1,
            phrase.features_30d.mean_f0_hz,
            phrase.features_30d.duration_ms,
            phrase.phrase_key
        );
    }
    println!();

    // ========================================================================
    // Step 3: Within-Call Analysis using Acoustic Similarity
    // ========================================================================
    println!("Step 3: Within-Call Analysis using Acoustic Similarity Engine...\n");

    let mut analyzer = WithinCallAnalyzer::for_species("marmoset");
    let result = analyzer.discover_phrases(phrases.clone(), "demo_call_001", "marmoset");

    println!("  === Discovered Phrase Types ===");
    println!("  Total phrases: {}", result.total_phrases);
    println!("  Unique types discovered: {}", result.unique_types);
    println!("  Type entropy: {:.3} bits", result.type_entropy);
    println!();

    for (i, pt) in result.phrase_types.iter().enumerate() {
        println!(
            "  Type {}: {} ({} occurrences)",
            i + 1,
            pt.type_id,
            pt.occurrence_count
        );
        println!("    Phrase key: {}", pt.phrase_key);
        println!("    Centroid F0: {:.0} Hz", pt.centroid_features.mean_f0_hz);
        println!(
            "    Centroid Duration: {:.0} ms",
            pt.centroid_features.duration_ms
        );
        println!("    Intra-type variability: {:.4}", pt.intra_variability);
    }
    println!();

    // ========================================================================
    // Step 4: Similarity Statistics
    // ========================================================================
    println!("Step 4: Similarity Statistics...\n");

    println!(
        "  Average within-type similarity: {:.4}",
        result.avg_within_type_similarity
    );
    println!(
        "  Average between-type distance: {:.4}",
        result.avg_between_type_distance
    );

    let separation = if result.avg_within_type_similarity > 0.0 {
        result.avg_between_type_distance / (1.0 - result.avg_within_type_similarity)
    } else {
        f64::INFINITY
    };
    println!("  Separation ratio: {:.2}x (higher = better)", separation);
    println!();

    // ========================================================================
    // Step 5: Phrase Sequence and Transitions
    // ========================================================================
    println!("Step 5: Phrase Sequence and Transitions...\n");

    println!(
        "  Phrase sequence: {} phrases total",
        result.phrase_sequence.len()
    );
    for (i, phrase_id) in result.phrase_sequence.iter().take(10).enumerate() {
        print!("  {} ", phrase_id.split('_').last().unwrap_or("?"));
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    println!("\n");

    println!("  Transition matrix:");
    for (from, transitions) in &result.transition_matrix {
        for (to, count) in transitions {
            println!("    {} → {}: {} occurrences", from, to, count);
        }
    }
    println!();

    // ========================================================================
    // Step 6: Motif Discovery
    // ========================================================================
    println!("Step 6: Motif Discovery...\n");

    let motifs = analyzer.find_motifs(&result, 2, 1);

    if motifs.is_empty() {
        println!("  No recurring motifs found (need more repetitions)");
    } else {
        println!("  Discovered {} motif(s):", motifs.len());
        for motif in &motifs {
            println!("    Pattern: {:?}", motif.pattern);
            println!("    Occurrences: {}", motif.occurrence_count);
            println!("    Positions: {:?}", motif.positions);
        }
    }
    println!();

    // ========================================================================
    // Step 7: Compare Standard vs Similarity-Based Library Building
    // ========================================================================
    println!("Step 7: Compare Library Building Approaches...\n");

    // Standard approach (key-based grouping)
    let standard_builder = ZooVoxLibraryBuilder::new().with_similarity_threshold(0.85);
    let standard_library = standard_builder.build_library(phrases.clone(), "marmoset", None)?;

    println!("  Standard (Key-Based):");
    println!("    Phrases: {}", standard_library.total_phrases);
    println!("    Occurrences: {}", standard_library.total_occurrences);
    println!("    Entropy: {:.3} bits", standard_library.type_entropy);

    // Similarity-based approach
    let similarity_builder = SimilarityBasedLibraryBuilder::for_species("marmoset");
    let similarity_library = similarity_builder.build_library(phrases.clone(), "marmoset")?;

    println!("\n  Similarity-Based:");
    println!("    Phrases: {}", similarity_library.total_phrases);
    println!("    Occurrences: {}", similarity_library.total_occurrences);
    println!("    Entropy: {:.3} bits", similarity_library.type_entropy);
    println!();

    // ========================================================================
    // Step 8: Species-Specific Configuration
    // ========================================================================
    println!("Step 8: Species-Specific Configuration Summary...\n");

    let species_configs = [
        ("sperm_whale", "Coda-type encoding", 0.90),
        ("dolphin", "FM whistle encoding", 0.80),
        ("zebra_finch", "Combinatorial encoding", 0.85),
        ("marmoset", "Harmonic encoding", 0.85),
    ];

    println!(
        "  {:<15} {:<25} {:<12}",
        "Species", "Encoding Type", "Sim Threshold"
    );
    println!("  {}", "-".repeat(55));

    for (species, encoding, threshold) in &species_configs {
        let config = WithinCallConfig::for_species(species);
        println!(
            "  {:<15} {:<25} {:.2}",
            species, encoding, config.similarity_threshold
        );
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("========================================");
    println!("Pipeline Summary");
    println!("========================================");
    println!("✓ Acoustic Similarity Engine integrated");
    println!("✓ Pairwise similarity-based phrase typing");
    println!("✓ Within-call phrase discovery");
    println!("✓ Transition matrix computation");
    println!("✓ Motif discovery support");
    println!("✓ Species-specific configurations");
    println!();
    println!("Key Differences from Standard Approach:");
    println!("  Standard: Groups by F0/Dur bins (discrete)");
    println!("  Similarity: Groups by acoustic similarity (continuous)");
    println!();
    println!("Advantages of Similarity-Based Approach:");
    println!("  - Better handles continuous acoustic variation");
    println!("  - Species-specific thresholds");
    println!("  - Provides intra-type variability metrics");
    println!("  - Supports k-NN classification for new phrases");

    Ok(())
}
