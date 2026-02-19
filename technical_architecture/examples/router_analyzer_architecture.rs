//! Router → Analyzer Architecture Demonstration
//!
//! This example demonstrates the correct use of species-specific weights:
//!
//! **Phase 1: Species Identification (Global Discrimination)**
//! - Uses UNIFIED weights or equal weights
//! - Compares across all species
//! - Goal: "What species is this?"
//!
//! **Phase 2: Phrase Analysis (Contextual Analysis)**
//! - Uses SPECIES-SPECIFIC weights
//! - Compares within a single species
//! - Goal: "What phrase type is this?"
//!
//! Key Insight from BEANS-Zero Benchmark:
//! ─────────────────────────────────────
//! Applying different weights per prototype breaks distance comparability.
//! When comparing a query to prototypes with different weight schemes,
//! the distances are in different "units" and cannot be meaningfully ranked.
//!
//! Solution: Use consistent weights for Phase 1, species-specific for Phase 2.

use technical_architecture::{
    species::FeatureWeights, zoo_vox_within_call::WithinCallAnalyzer, AcousticSimilarityEngine,
    SimilarityMetric,
};

const FEATURE_DIM: usize = 45;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Router → Analyzer Architecture Demo                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // PHASE 1: Global Species Identification
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: Global Species Identification                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Uses UNIFIED weights for cross-species comparison                          │");
    println!("│ Goal: 'What species is this?'                                              │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create engine with UNIFIED weights (or no weights = equal)
    let mut global_engine =
        AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Unified weights for global discrimination
    let unified_weights = FeatureWeights {
        spectral: 1.2,
        harmonic: 1.0,
        temporal: 1.2,
        modulation: 1.2,
        cepstral: 1.0,
        formant: 1.0,
        micro_dynamics: 1.0,
        psychoacoustic: 1.0,
        tfs: 1.0,
        overrides: vec![],
    };

    global_engine.set_feature_weights(&unified_weights.to_weight_vector());

    println!("Applied unified weights:");
    println!("  └─ All feature groups: 1.0-1.2x (balanced across species)");
    println!();

    // Simulate comparing unknown sample to species prototypes
    let unknown_sample = vec![0.5; FEATURE_DIM]; // Placeholder

    let dolphin_prototype = vec![0.6; FEATURE_DIM];
    let finch_prototype = vec![0.4; FEATURE_DIM];
    let bat_prototype = vec![0.7; FEATURE_DIM];

    let dist_dolphin = global_engine.distance(
        &ndarray::Array1::from_vec(unknown_sample.clone()),
        &ndarray::Array1::from_vec(dolphin_prototype),
    );
    let dist_finch = global_engine.distance(
        &ndarray::Array1::from_vec(unknown_sample.clone()),
        &ndarray::Array1::from_vec(finch_prototype),
    );
    let dist_bat = global_engine.distance(
        &ndarray::Array1::from_vec(unknown_sample.clone()),
        &ndarray::Array1::from_vec(bat_prototype),
    );

    println!("Cross-species distances (using unified weights):");
    println!("  ├─ Dolphin: {:.4}", dist_dolphin);
    println!("  ├─ Zebra Finch: {:.4}", dist_finch);
    println!("  └─ Egyptian Bat: {:.4}", dist_bat);
    println!();

    // Identify species
    let species_distances = [
        ("Dolphin", dist_dolphin),
        ("Zebra Finch", dist_finch),
        ("Egyptian Bat", dist_bat),
    ];
    let (species, _) = species_distances
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("→ Identified Species: {}", species);
    println!();

    // =========================================================================
    // PHASE 2: Within-Species Phrase Analysis
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: Within-Species Phrase Analysis                                     │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Uses SPECIES-SPECIFIC weights for within-species discrimination            │");
    println!("│ Goal: 'What phrase type is this?'                                           │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create analyzer with Dolphin-specific weights
    // This is the CORRECT use of species weights!
    let _dolphin_analyzer = WithinCallAnalyzer::for_species("dolphin");

    println!("Applied Dolphin-specific weights:");
    let dolphin_weights = FeatureWeights::dolphin();
    println!(
        "  ├─ Modulation (FM slope): {:.1}x ← CRITICAL for whistles",
        dolphin_weights.modulation
    );
    println!("  ├─ Spectral: {:.1}x", dolphin_weights.spectral);
    println!(
        "  └─ Micro-dynamics: {:.1}x (long contours, less micro-structure)",
        dolphin_weights.micro_dynamics
    );
    println!();

    // Create analyzers for other species
    let _finch_analyzer = WithinCallAnalyzer::for_species("zebra_finch");
    let finch_weights = FeatureWeights::zebra_finch();
    println!("Applied Zebra Finch-specific weights:");
    println!(
        "  ├─ Harmonic: {:.1}x ← Harmonic stack structure in song",
        finch_weights.harmonic
    );
    println!(
        "  ├─ Temporal: {:.1}x ← Syllable timing important",
        finch_weights.temporal
    );
    println!(
        "  └─ Micro-dynamics: {:.1}x ← Syllable transitions",
        finch_weights.micro_dynamics
    );
    println!();

    let _whale_analyzer = WithinCallAnalyzer::for_species("sperm_whale");
    let whale_weights = FeatureWeights::sperm_whale();
    println!("Applied Sperm Whale-specific weights:");
    println!(
        "  ├─ Temporal: {:.1}x ← TIMING IS EVERYTHING for codas",
        whale_weights.temporal
    );
    println!(
        "  ├─ Micro-dynamics: {:.1}x ← Click patterns",
        whale_weights.micro_dynamics
    );
    println!(
        "  └─ TFS: {:.1}x ← Temporal structure critical",
        whale_weights.tfs
    );
    println!();

    // =========================================================================
    // KEY ARCHITECTURAL LESSON
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("KEY ARCHITECTURAL LESSON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("The BEANS-Zero benchmark proved that species-specific weights CANNOT be");
    println!("used in global k-NN search. The results:");
    println!();
    println!("  Unweighted F1:      47.83%");
    println!("  Species-Aware F1:   29.71%  ← -37.88% degradation!");
    println!();
    println!("WHY: When each prototype has different weights, distances become");
    println!("non-comparable (like measuring some distances in inches and others in cm).");
    println!();
    println!("SOLUTION: Router → Analyzer architecture:");
    println!("  1. Phase 1 uses UNIFIED weights for cross-species comparison");
    println!("  2. Phase 2 uses SPECIES-SPECIFIC weights for within-species analysis");
    println!();
    println!("This is now implemented in WithinCallAnalyzer::for_species()");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
