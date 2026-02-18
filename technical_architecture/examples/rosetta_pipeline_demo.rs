//! Rosetta Pipeline Demonstration
//!
//! This example demonstrates the complete Zoo Vox Rosetta Pipeline:
//!
//! **Phase 1: Global Species Identification**
//! - Uses unified weights for cross-species comparison
//!
//! **Phase 2a: Semantic Grounding (Human-Guided)**
//! - Matches phrases against pre-seeded semantic dictionaries
//!
//! **Phase 2b: Contextual Enrichment (Environmental/Syntax)**
//! - Refines interpretation based on sensor data

use technical_architecture::{
    rosetta_pipeline::{
        RosettaPipeline, RosettaBundle, RosettaResult,
        SemanticPhraseDictionary, ContextEnrichedPhrase,
        EnvState, FEATURE_DIM,
    },
    species::FeatureWeights,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Zoo Vox Rosetta Pipeline Demonstration                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // STEP 1: Create Semantic Dictionaries (from Human-Guided Discovery)
    // =========================================================================
    println!("[1/4] Creating semantic dictionaries from human annotations...");
    println!();

    // Marmoset dictionary (from Human-Guided Context Discovery)
    let mut marmoset_type_to_labels = HashMap::new();

    // Type_53: Phee calls (100% from annotations)
    let mut type53_labels = HashMap::new();
    type53_labels.insert("Phee".to_string(), 1.0);
    marmoset_type_to_labels.insert("Type_53".to_string(), type53_labels);

    // Type_300: Twitter calls (50% from annotations)
    let mut type300_labels = HashMap::new();
    type300_labels.insert("Twitter".to_string(), 0.5);
    type300_labels.insert("Vocalization".to_string(), 0.5);
    marmoset_type_to_labels.insert("Type_300".to_string(), type300_labels);

    // Type_16: General vocalizations (87% from annotations)
    let mut type16_labels = HashMap::new();
    type16_labels.insert("Vocalization".to_string(), 0.87);
    type16_labels.insert("Tsik".to_string(), 0.13);
    marmoset_type_to_labels.insert("Type_16".to_string(), type16_labels);

    // Create centroids (simplified - in practice these come from feature extraction)
    let mut marmoset_centroids = HashMap::new();
    marmoset_centroids.insert("Type_53".to_string(), vec![0.5; FEATURE_DIM]);
    marmoset_centroids.insert("Type_300".to_string(), vec![0.4; FEATURE_DIM]);
    marmoset_centroids.insert("Type_16".to_string(), vec![0.6; FEATURE_DIM]);

    let marmoset_dict = SemanticPhraseDictionary {
        species: "marmoset".to_string(),
        type_to_labels: marmoset_type_to_labels,
        type_centroids: marmoset_centroids,
        total_phrases: 5000,
        num_types: 3700,
    };

    // Egyptian Fruit Bat dictionary
    let mut bat_type_to_labels = HashMap::new();

    // Type_52: Fighting calls (42% from annotations)
    let mut type52_labels = HashMap::new();
    type52_labels.insert("Fighting".to_string(), 0.42);
    type52_labels.insert("Unknown".to_string(), 0.58);
    bat_type_to_labels.insert("Type_52".to_string(), type52_labels);

    let mut bat_centroids = HashMap::new();
    bat_centroids.insert("Type_52".to_string(), vec![0.7; FEATURE_DIM]);

    let bat_dict = SemanticPhraseDictionary {
        species: "egyptian_fruit_bat".to_string(),
        type_to_labels: bat_type_to_labels,
        type_centroids: bat_centroids,
        total_phrases: 2000,
        num_types: 768,
    };

    println!("  ├─ Marmoset: {} types from {} phrases", marmoset_dict.num_types, marmoset_dict.total_phrases);
    println!("  └─ Egyptian Fruit Bat: {} types from {} phrases", bat_dict.num_types, bat_dict.total_phrases);
    println!();

    // =========================================================================
    // STEP 2: Create Rosetta Bundles
    // =========================================================================
    println!("[2/4] Creating Rosetta Bundles...");
    println!();

    // Create marmoset bundle
    let marmoset_bundle = RosettaBundle::new(
        "marmoset",
        FeatureWeights::marmoset(),
        marmoset_dict,
        FeatureWeights::unified(),
    );

    // Create bat bundle
    let bat_bundle = RosettaBundle::new(
        "egyptian_fruit_bat",
        FeatureWeights::bat(),
        bat_dict,
        FeatureWeights::unified(),
    );

    println!("  Marmoset Bundle:");
    println!("    ├─ Version: {}", marmoset_bundle.version);
    println!("    ├─ Species: {}", marmoset_bundle.species);
    println!("    ├─ Feature weights: marmoset-specific");
    println!("    └─ Dictionary: {} phrase types", marmoset_bundle.semantic_dictionary.num_types);
    println!();

    println!("  Egyptian Fruit Bat Bundle:");
    println!("    ├─ Version: {}", bat_bundle.version);
    println!("    ├─ Species: {}", bat_bundle.species);
    println!("    ├─ Feature weights: bat-specific");
    println!("    └─ Dictionary: {} phrase types", bat_bundle.semantic_dictionary.num_types);
    println!();

    // =========================================================================
    // STEP 3: Initialize Pipeline and Load Bundles
    // =========================================================================
    println!("[3/4] Initializing Rosetta Pipeline...");
    println!();

    let mut pipeline = RosettaPipeline::new()?;
    pipeline.load_bundle(marmoset_bundle);
    pipeline.load_bundle(bat_bundle);

    println!("  Loaded species:");
    for species in pipeline.loaded_species() {
        println!("    └─ {}", species);
    }
    println!();

    // =========================================================================
    // STEP 4: Process Audio Streams
    // =========================================================================
    println!("[4/4] Processing audio streams...");
    println!();

    // Simulate processing different audio streams
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ STREAM 1: Marmoset Audio (Quiet Environment)                                 │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let marmoset_audio: Vec<f32> = vec![0.1; 4800]; // 100ms at 48kHz
    let result = pipeline.process_stream(&marmoset_audio, EnvState::Quiet)?;
    print_result(&result);

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ STREAM 2: Marmoset Audio (Windy Environment)                                 │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let result = pipeline.process_stream(&marmoset_audio, EnvState::Wind)?;
    print_result(&result);

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ STREAM 3: Bat Audio (Storm Environment)                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let bat_audio: Vec<f32> = vec![0.3; 4800]; // 100ms at 48kHz
    let result = pipeline.process_stream(&bat_audio, EnvState::Storm)?;
    print_result(&result);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("The Rosetta Pipeline integrates:");
    println!();
    println!("  Phase 1: Global Species Identification");
    println!("    ├─ Uses unified weights for cross-species comparison");
    println!("    └─ Identifies species from acoustic features");
    println!();
    println!("  Phase 2a: Semantic Grounding (Human-Guided)");
    println!("    ├─ Matches phrases against pre-seeded semantic dictionaries");
    println!("    └─ Maps acoustic types to semantic labels (e.g., \"Type_53\" → \"Phee\")");
    println!();
    println!("  Phase 2b: Contextual Enrichment (Environmental/Syntax)");
    println!("    ├─ Refines interpretation based on sensor data");
    println!("    └─ Infers intent from semantic label + environment");
    println!();
    println!("  Output: ContextEnrichedPhrase");
    println!("    ├─ semantic_label: \"Phee\" (from human annotations)");
    println!("    ├─ environmental_state: \"Windy\" (from sensors)");
    println!("    └─ inferred_intent: \"Long_Range_Contact\" (combined)");
    println!();

    Ok(())
}

fn print_result(result: &RosettaResult) {
    println!("│ Species: {} (confidence: {:.1}%)", result.species, result.species_confidence * 100.0);
    println!("│ Processing time: {:.2}ms", result.processing_time_ms);
    println!("│ Phrases detected: {}", result.phrases.len());

    for phrase in &result.phrases {
        println!("│");
        println!("│ ┌─ Phrase: {}", phrase.phrase_type_id);
        println!("│ │  Semantic Label: {} ({:.0}% confidence)",
            phrase.semantic_label, phrase.label_confidence * 100.0);
        println!("│ │  Grading Score: {:.2} (0=discrete, 1=graded)", phrase.grading_score);
        println!("│ │  Environment: {:?}", phrase.environmental_state);
        println!("│ └─ Inferred Intent: {}", phrase.inferred_intent);
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
}
