// Corpus Analysis: Marmoset "Phrase X" Discovery
// =================================================
//
// This example demonstrates Phrase X discovery on real marmoset vocalization data.
// It loads the marmoset corpus and discovers linguistic units with:
// - Rigid internal structure (high PMI)
// - Flexible external connections (high suffix entropy)

use std::fs;
use std::path::Path;
use technical_architecture::{
    CorpusStatistics, NGramMiner, PhraseXDiscoveryEngine, PMICalculator,
    SuffixEntropyCalculator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Corpus Analysis: Phrase X Discovery (Marmoset)                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // Step 1: Load Marmoset Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Marmoset Corpus                                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let corpus_path = "/mnt/c/Users/sheel/Desktop/src/marmoset_corpus_for_analysis.json";

    if !Path::new(corpus_path).exists() {
        println!("❌ Corpus file not found: {}", corpus_path);
        println!("   Run the Python script first to generate the corpus data.");
        return Err("Corpus file not found".into());
    }

    println!("📂 Loading corpus from: {}", corpus_path);
    let corpus_content = fs::read_to_string(corpus_path)?;
    let corpus_data: serde_json::Value = serde_json::from_str(&corpus_content)?;

    // Extract sessions (cluster ID sequences)
    let sessions: Vec<Vec<usize>> = corpus_data["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            s.as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();

    let cluster_to_phrase: std::collections::HashMap<usize, String> = corpus_data["cluster_to_phrase"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse::<usize>().unwrap(), v.as_str().unwrap().to_string()))
        .collect();

    println!("✅ Corpus loaded successfully");
    println!("   Sessions: {}", sessions.len());
    println!("   Total phrases: {}", sessions.iter().map(|s| s.len()).sum::<usize>());
    println!("   Vocabulary size: {}", cluster_to_phrase.len());
    println!();

    // ========================================================================
    // Step 2: Calculate Corpus Statistics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Corpus Statistics                                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let stats = CorpusStatistics::from_corpus(&sessions)?;
    println!("Corpus Statistics:");
    println!("  Total sequences: {}", stats.total_sequences);
    println!("  Total symbols: {}", stats.total_symbols);
    println!("  Vocabulary size: {}", stats.vocabulary_size);
    println!("  Avg sequence length: {:.1} symbols", stats.avg_sequence_length);
    println!("  Unique n-grams: {}", stats.unique_ngrams);
    println!();

    // ========================================================================
    // Step 3: Extract N-Grams
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: N-Gram Extraction                                               │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let miner = NGramMiner::default();
    let ngram_counts = miner.count_ngrams(&sessions);

    println!("Top 15 Most Frequent N-grams:");
    let mut sorted_ngrams: Vec<_> = ngram_counts.iter().collect();
    sorted_ngrams.sort_by(|a, b| b.1.cmp(a.1));

    for (i, (ngram, count)) in sorted_ngrams.iter().take(15).enumerate() {
        // Convert cluster IDs to phrase names
        let phrase_names: Vec<String> = ngram.symbols.iter()
            .map(|id| cluster_to_phrase.get(id)
                .and_then(|s| {
                    // Parse F0 from phrase key
                    if s.starts_with("F0_") {
                        let rest = &s[3..];
                        if let Some(idx) = rest.find("_DUR_") {
                            Some(format!("F0={}", &rest[..idx]))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("ID{}", id)))
            .collect();

        println!("  {:>2}. {:30} | freq: {:>3}",
                 i + 1,
                 phrase_names.join(" → "),
                 count);
    }
    println!();

    // ========================================================================
    // Step 4: Calculate PMI (Internal Rigidity)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Internal Rigidity (PMI)                                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let pmi_calculator = PMICalculator::from_corpus(&sessions)?;

    println!("Internal Rigidity Scores (Top 15):");
    for (i, (ngram, _count)) in sorted_ngrams.iter().take(15).enumerate() {
        let pmi = pmi_calculator.average_pmi(ngram).unwrap_or(0.0);
        let phrase_names: Vec<String> = ngram.symbols.iter()
            .map(|id| cluster_to_phrase.get(id)
                .and_then(|s| {
                    if s.starts_with("F0_") {
                        let rest = &s[3..];
                        if let Some(idx) = rest.find("_DUR_") {
                            Some(format!("F0={}", &rest[..idx]))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("ID{}", id)))
            .collect();

        println!("  {:>2}. {:30} | PMI: {:.3}",
                 i + 1,
                 phrase_names.join(" → "),
                 pmi);
    }
    println!();

    // ========================================================================
    // Step 5: Calculate Suffix Entropy (External Flexibility)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: External Flexibility (Suffix Entropy)                          │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let entropy_calculator = SuffixEntropyCalculator::from_corpus(&sessions)?;

    println!("External Flexibility Scores (Top 15):");
    for (i, (ngram, _count)) in sorted_ngrams.iter().take(15).enumerate() {
        let entropy = entropy_calculator.suffix_entropy(ngram);
        let suffix_dist = entropy_calculator.suffix_distribution(ngram);

        let phrase_names: Vec<String> = ngram.symbols.iter()
            .map(|id| cluster_to_phrase.get(id)
                .and_then(|s| {
                    if s.starts_with("F0_") {
                        let rest = &s[3..];
                        if let Some(idx) = rest.find("_DUR_") {
                            Some(format!("F0={}", &rest[..idx]))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("ID{}", id)))
            .collect();

        println!("  {:>2}. {:30} | Entropy: {:.3} | Suffixes: {}",
                 i + 1,
                 phrase_names.join(" → "),
                 entropy,
                 suffix_dist.len());
    }
    println!();

    // ========================================================================
    // Step 6: Discover Phrase X Units
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Phrase X Discovery                                             │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Use adaptive thresholds based on corpus statistics
    let min_freq = if stats.total_symbols > 1000 { 3 } else { 2 };

    println!("Phrase X Discovery Parameters:");
    println!("  Min frequency: {}", min_freq);
    println!("  Rigidity threshold (PMI): 0.5");
    println!("  Flexibility threshold (Entropy): 0.5");
    println!();

    let engine = PhraseXDiscoveryEngine::new(&sessions, min_freq, 0.5, 0.5)?;
    let phrases = engine.discover()?;

    println!("Total phrase candidates discovered: {}", phrases.len());
    println!();

    let phrases_x = engine.filter_phrases_x(&phrases);

    if phrases_x.is_empty() {
        println!("⚠️  No Phrase X candidates found with current thresholds.");
        println!();
        println!("Top 10 phrase candidates:");
        for (i, phrase) in phrases.iter().take(10).enumerate() {
            let phrase_names: Vec<String> = phrase.ngram.symbols.iter()
                .map(|id| cluster_to_phrase.get(id)
                    .and_then(|s| {
                        if s.starts_with("F0_") {
                            let rest = &s[3..];
                            if let Some(idx) = rest.find("_DUR_") {
                                Some(format!("F0={}", &rest[..idx]))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| format!("ID{}", id)))
                .collect();

            println!("  {}. {:30} | Rigidity: {:.3} | Flexibility: {:.3} | Freq: {}",
                     i + 1,
                     phrase_names.join(" → "),
                     phrase.rigidity_score,
                     phrase.flexibility_score,
                     phrase.frequency);
        }
    } else {
        println!("✅ Found {} Phrase X candidates:", phrases_x.len());
        println!();

        for (i, phrase) in phrases_x.iter().take(20).enumerate() {
            let phrase_names: Vec<String> = phrase.ngram.symbols.iter()
                .map(|id| cluster_to_phrase.get(id)
                    .and_then(|s| {
                        if s.starts_with("F0_") {
                            let rest = &s[3..];
                            if let Some(idx) = rest.find("_DUR_") {
                                Some(format!("F0={}", &rest[..idx]))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| format!("ID{}", id)))
                .collect();

            println!("  {}. Phrase: {}", i + 1, phrase_names.join(" → "));
            println!("     Rigidity: {:.3} | Flexibility: {:.3} | Frequency: {}",
                     phrase.rigidity_score,
                     phrase.flexibility_score,
                     phrase.frequency);
            println!("     Suffix diversity: {} different following symbols",
                     phrase.suffix_diversity());

            // Show example contexts
            let contexts = engine.analyze_context_variability(phrase, &sessions);
            if !contexts.is_empty() {
                println!("     Example contexts (showing {}):", contexts.len().min(3));
                for (j, ctx) in contexts.iter().take(3).enumerate() {
                    let ctx_names: Vec<String> = ctx.iter()
                        .map(|id| cluster_to_phrase.get(id)
                            .and_then(|s| {
                                if s.starts_with("F0_") {
                                    let rest = &s[3..];
                                    if let Some(idx) = rest.find("_DUR_") {
                                        Some(format!("F0={}", &rest[..idx]))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| format!("ID{}", id)))
                        .collect();
                    println!("       {}: {:?}", j + 1, ctx_names);
                }
            }
            println!();
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                        ANALYSIS COMPLETE                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Summary:");
    println!("  • Corpus size: {} sessions, {} total phrases",
             sessions.len(),
             sessions.iter().map(|s| s.len()).sum::<usize>());
    println!("  • Vocabulary: {} unique phrase types", cluster_to_phrase.len());
    println!("  • Phrase X candidates: {}", phrases_x.len());
    println!();

    if !phrases_x.is_empty() {
        println!("Interpretation:");
        println!("  → Phrase X units are linguistically meaningful");
        println!("  → Internal structure is rigid (high PMI)");
        println!("  → External connections are flexible (high entropy)");
        println!("  → These may represent: functional words, calls, or phrases");
    }

    Ok(())
}
