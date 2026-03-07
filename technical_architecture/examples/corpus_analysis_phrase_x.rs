// Corpus Analysis: Discovering "Phrase X" Units
// ================================================
//
// This example demonstrates how to use the corpus analysis module to discover
// "Phrase X" units - linguistic units with rigid internal structure but
// flexible external connections.
//
// The algorithm uses:
// 1. PMI (Pointwise Mutual Information) to measure internal rigidity
// 2. Suffix Entropy to measure external flexibility
//
// Reference: Universal Rosetta Stone methodology for cross-species
// communication analysis.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use technical_architecture::{
    CorpusStatistics, NGram, NGramMiner, PMICalculator, PhraseX, PhraseXDiscoveryEngine, SuffixEntropyCalculator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Corpus Analysis: Phrase X Discovery ===\n");

    // ========================================================================
    // Step 1: Create a sample corpus
    // ========================================================================
    //
    // In a real scenario, this would be the output of your clustering pipeline:
    // - Extract 30D features from audio
    // - Cluster using DTW-DBSCAN or HDBSCAN
    // - Output: Sequence of Cluster IDs
    //
    // For this example, we use symbolic sequences where:
    // - 101, 102, 103... represent different "words" or "syllables" discovered by clustering
    // - The corpus represents multiple vocalization sessions

    let corpus = vec![
        // Session 1: [101, 102] is a potential "Phrase X"
        vec![101, 102, 201, 101, 102, 202, 101, 102, 203],
        // Session 2: [301, 302] is a "rigid chain" (always followed by 401)
        vec![301, 302, 401, 301, 302, 401, 301, 302, 401],
        // Session 3: More [101, 102] with different suffixes
        vec![101, 102, 204, 501, 502, 101, 102, 205],
        // Session 4: Mixed patterns
        vec![101, 102, 206, 301, 302, 401, 101, 102, 207],
        // Session 5: [101, 102] followed by yet another symbol
        vec![601, 101, 102, 208, 701, 702],
    ];

    println!("Corpus contains {} sessions", corpus.len());
    for (i, session) in corpus.iter().enumerate() {
        println!("  Session {}: {:?}", i + 1, session);
    }
    println!();

    // ========================================================================
    // Step 2: Calculate corpus statistics
    // ========================================================================

    let stats = CorpusStatistics::from_corpus(&corpus)?;
    println!("=== Corpus Statistics ===");
    println!("  Total sequences: {}", stats.total_sequences);
    println!("  Total symbols: {}", stats.total_symbols);
    println!("  Vocabulary size: {} unique symbols", stats.vocabulary_size);
    println!("  Avg sequence length: {:.1} symbols", stats.avg_sequence_length);
    println!();

    // ========================================================================
    // Step 3: Extract N-grams
    // ========================================================================

    let miner = NGramMiner::default();
    let ngram_counts = miner.count_ngrams(&corpus);

    println!("=== Top 5 Most Frequent N-grams ===");
    let mut sorted_ngrams: Vec<_> = ngram_counts.iter().collect();
    sorted_ngrams.sort_by(|a, b| b.1.cmp(a.1));

    for (i, (ngram, count)) in sorted_ngrams.iter().take(5).enumerate() {
        println!("  {}. {} (frequency: {})", i + 1, ngram, count);
    }
    println!();

    // ========================================================================
    // Step 4: Calculate PMI (Internal Rigidity)
    // ========================================================================

    let pmi_calculator = PMICalculator::from_corpus(&corpus)?;

    println!("=== Internal Rigidity (PMI Scores) ===");
    // PMI measures how strongly two symbols are associated
    // High PMI = symbols tend to appear together (rigid internal structure)

    for (ngram, _count) in sorted_ngrams.iter().take(5) {
        let pmi = pmi_calculator.average_pmi(ngram).unwrap_or(0.0);
        println!("  {}: PMI = {:.3}", ngram, pmi);
    }
    println!();

    // ========================================================================
    // Step 5: Calculate Suffix Entropy (External Flexibility)
    // ========================================================================

    let entropy_calculator = SuffixEntropyCalculator::from_corpus(&corpus)?;

    println!("=== External Flexibility (Suffix Entropy) ===");
    // Entropy measures how predictable the following symbol is
    // High entropy = many different following symbols (flexible external connections)
    // Low entropy = always follows the same pattern (rigid chain)

    for (ngram, _count) in sorted_ngrams.iter().take(5) {
        let entropy = entropy_calculator.suffix_entropy(ngram);
        let suffix_dist = entropy_calculator.suffix_distribution(ngram);

        println!("  {}: Entropy = {:.3}", ngram, entropy);
        if !suffix_dist.is_empty() {
            println!("     Followed by: {:?}", suffix_dist);
        }
    }
    println!();

    // ========================================================================
    // Step 6: Discover Phrase X Units
    // ========================================================================

    println!("=== Phrase X Discovery ===");
    // A "Phrase X" has:
    // - High internal rigidity (high PMI)
    // - High external flexibility (high entropy)

    // Use lower thresholds for this small example corpus
    let engine = PhraseXDiscoveryEngine::new(&corpus, 2, 0.1, 0.1)?;
    let phrases = engine.discover()?;
    let phrases_x = engine.filter_phrases_x(&phrases);

    if phrases_x.is_empty() {
        println!("  No Phrase X candidates found with current thresholds.");
        println!("  Top phrase candidates:");
        for (i, phrase) in phrases.iter().take(5).enumerate() {
            println!(
                "  {}. Rigidity={:.3}, Flexibility={:.3}, Freq={}",
                i + 1,
                phrase.rigidity_score,
                phrase.flexibility_score,
                phrase.frequency
            );
        }
    } else {
        println!("  Found {} Phrase X candidates:", phrases_x.len());
        for (i, phrase) in phrases_x.iter().enumerate() {
            println!("  {}. {}", i + 1, phrase);
            println!(
                "     Rigidity: {:.3}, Flexibility: {:.3}, Frequency: {}",
                phrase.rigidity_score, phrase.flexibility_score, phrase.frequency
            );
            println!(
                "     Suffix diversity: {} different following symbols",
                phrase.suffix_diversity()
            );

            // Show context variability
            let contexts = engine.analyze_context_variability(phrase, &corpus);
            println!("     Example contexts (showing {}):", contexts.len().min(3));
            for (j, ctx) in contexts.iter().take(3).enumerate() {
                println!("       {}: {:?}", j + 1, ctx);
            }
            println!();
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================

    println!("=== Summary ===");
    println!("The corpus analysis pipeline successfully identifies:");
    println!("1. Rigid internal structure (high PMI)");
    println!("2. Flexible external connections (high entropy)");
    println!();
    println!("In this example:");
    println!("- [101, 102] appears with high diversity of following symbols");
    println!("- [301, 302] always followed by 401 (low entropy = rigid chain)");
    println!();
    println!("For real animal vocalization analysis:");
    println!("1. Run your clustering pipeline on audio data");
    println!("2. Collect Cluster ID sequences from all recordings");
    println!("3. Use PhraseXDiscoveryEngine to find meaningful units");
    println!("4. Analyze context variability for each discovered phrase");

    Ok(())
}

// =============================================================================
// Helper Function to Create Realistic Test Corpus
// =============================================================================

#[allow(dead_code)]
fn create_realistic_test_corpus() -> Vec<Vec<usize>> {
    // Simulates a realistic scenario where:
    // - IDs 100-199: "Food" related vocalizations
    // - IDs 200-299: "Danger" related vocalizations
    // - IDs 300-399: "Social" related vocalizations
    // - IDs 400-499: "Mating" related vocalizations
    //
    // Phrase X: [101, 102] = "Food request" (rigid, followed by many different responses)
    // Rigid chain: [301, 302, 303] = "Greeting sequence" (always same pattern)

    vec![
        // Food requests with varied responses
        vec![101, 102, 201, 101, 102, 202, 101, 102, 203],
        vec![101, 102, 204, 501, 502, 101, 102, 205, 101, 102, 206],
        vec![101, 102, 207, 301, 302, 303, 101, 102, 208],
        // Greeting sequences (rigid pattern)
        vec![301, 302, 303, 401, 301, 302, 303, 401, 301, 302, 303, 401],
        vec![301, 302, 303, 401, 301, 302, 303, 402, 301, 302, 303, 401],
        // Mixed vocalizations
        vec![601, 602, 101, 102, 209, 701, 301, 302, 303, 401],
        vec![101, 102, 210, 801, 802, 803, 101, 102, 211, 901],
    ]
}
