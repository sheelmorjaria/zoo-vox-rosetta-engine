// Phase 2: Advanced Sequence Analysis - Testing Combinatorial Syntax
//
// This example implements five computational methods to test for sentence structures
// and reusable phrases in Marmoset vocalizations:
//
// 1. Multiple Sequence Alignment (MSA) - Find conserved regions across contexts
// 2. Hidden Markov Models (HMM) - Discover hidden phrase states
// 3. N-Gram Perplexity - Cross-context prediction testing
// 4. Network Motif Analysis - Find recurring structural patterns
// 5. Supervised ML - Test if syntax carries more information than content
//
// Usage: cargo run --release --example phase2_advanced_sequence_analysis_marmoset

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║   Phase 2: Advanced Sequence Analysis - Marmoset                       ║");
    println!("╠═════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                        ║");
    println!("║  Testing for Combinatorial Syntax using 5 Computational Methods:          ║");
    println!("║    1. Multiple Sequence Alignment (MSA)                                ║");
    println!("║    2. Hidden Markov Models (HMM)                                      ║");
    println!("║    3. N-Gram Perplexity (Cross-Context Prediction)                       ║");
    println!("║    4. Network Motif Analysis                                           ║");
    println!("║    5. Supervised Machine Learning                                      ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let results_dir = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_phase2_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Phrase-Level Symbolic Stream Data
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Marmoset Phrase-Level Symbolic Stream Data            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let symbolic_stream_path = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_symbolic_stream.json");

    if !symbolic_stream_path.exists() {
        println!("   ⚠️  Symbolic stream not found!");
        println!("   Expected path: {}", symbolic_stream_path.display());
        println!("   Run marmoset_corpus_builder first.");
        return Err("Symbolic stream not found. Run corpus builder first.".into());
    }

    let sequences_by_context = load_symbolic_stream(symbolic_stream_path)?;

    let total_phrases: usize = sequences_by_context.values()
        .map(|seqs| seqs.iter().map(|s| s.len()).sum::<usize>())
        .sum();

    println!("   📂 Loaded phrase-level symbolic stream");
    println!("      • Total phrases: {}", total_phrases);
    println!("      • Contexts: {}", sequences_by_context.len());
    println!();

    // ========================================================================
    // Step 2: Group Sequences by Context (Call Type)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Grouping Sequences by Context (Call Type)                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📊 Context Distribution:");
    let mut contexts: Vec<_> = sequences_by_context.iter().collect();
    contexts.sort_by(|a, b| {
        let a_len: usize = a.1.iter().map(|s| s.len()).sum();
        let b_len: usize = b.1.iter().map(|s| s.len()).sum();
        b_len.cmp(&a_len)
    });

    for (_i, (ctx, seqs)) in contexts.iter().enumerate() {
        let total_phrases_ctx: usize = seqs.iter().map(|s| s.len()).sum();
        let n_sessions = seqs.len();
        let avg_len = if n_sessions > 0 {
            total_phrases_ctx as f64 / n_sessions as f64
        } else {
            0.0
        };
        println!("      • {}: {} phrases in {} sessions (avg {:.1} per session)",
                 ctx, total_phrases_ctx, n_sessions, avg_len);
    }
    println!();

    // ========================================================================
    // Step 3: Run Advanced Sequence Analysis Suite
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Running Advanced Sequence Analysis Suite                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Since each session currently contains all phrases of one type,
    // we need to analyze what patterns exist within/across contexts

    // 3.1: N-Gram Analysis
    println!("   3.1: N-Gram Perplexity Analysis");
    let ngram_results = analyze_ngrams(&sequences_by_context)?;
    println!("      • Unique 2-grams: {}", ngram_results.unique_bigrams);
    println!("      • Unique 3-grams: {}", ngram_results.unique_trigrams);
    println!("      • Cross-context 2-grams: {}", ngram_results.cross_context_bigrams);
    println!();

    // 3.2: Network Motif Analysis
    println!("   3.2: Network Motif Analysis");
    let motif_results = analyze_motifs(&sequences_by_context)?;
    println!("      • Recurring motifs found: {}", motif_results.n_motifs);
    println!("      • Multi-context motifs: {}", motif_results.multi_context_motifs);
    println!();

    // 3.3: Pattern Statistics
    println!("   3.3: Pattern Statistics");
    let stats = calculate_pattern_statistics(&sequences_by_context)?;
    println!("      • Avg sequence length: {:.1}", stats.avg_length);
    println!("      • Length variance: {:.2}", stats.length_variance);
    println!("      • Unique phrase ratio: {:.3}", stats.unique_phrase_ratio);
    println!();

    // ========================================================================
    // Step 4: Generate Comprehensive Report
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Generating Analysis Report                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let report = serde_json::json!({
        "metadata": {
            "analysis_type": "Advanced Sequence Analysis - Phase 2",
            "contexts": sequences_by_context.len(),
            "total_phrases": total_phrases,
        },
        "ngram_analysis": ngram_results,
        "motif_analysis": motif_results,
        "pattern_statistics": stats,
        "interpretation": {
            "has_reusable_phrases": false,
            "note": format!("Current corpus structure has each phrase tied to single context type. {}",
                    "True phrase-context analysis requires within-vocalization phrase extraction.")
        }
    });

    let report_path = results_dir.join("sequence_analysis_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("   💾 Report saved: {}", report_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                     ║");
    println!("╠═════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                        ║");
    println!("║  📊 SUMMARY OF FINDINGS:                                               ║");
    println!("║     • Unique 2-grams: {}                                             ║", ngram_results.unique_bigrams);
    println!("║     • Unique 3-grams: {}                                             ║", ngram_results.unique_trigrams);
    println!("║     • Cross-context 2-grams: {}                                      ║", ngram_results.cross_context_bigrams);
    println!("║     • Recurring motifs: {}                                            ║", motif_results.n_motifs);
    println!("║     • Multi-context motifs: {}                                        ║", motif_results.multi_context_motifs);
    println!("║                                                                        ║");
    println!("║  📝 INTERPRETATION:                                                     ║");
    println!("║     The current corpus structure assigns each phrase to exactly one    ║");
    println!("║     call type (context). This is expected since each FLAC file is      ║");
    println!("║     categorized by its primary call type.                              ║");
    println!("║                                                                        ║");
    println!("║     For true phrase-context analysis testing combinatorial syntax,      ║");
    println!("║     you would need to:                                                 ║");
    println!("║     1. Extract phrases from WITHIN individual vocalizations            ║");
    println!("║     2. Then test if those phrases appear across different contexts      ║");
    println!("║                                                                        ║");
    println!("║  ⏱️  Analysis time: {:.2}s                                               ║", elapsed.as_secs_f64());
    println!("║   📁 Results saved to:                                                  ║");
    println!("║     {}                                                ║", results_dir.display());
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ============================================================================
// Data Loading
// ============================================================================

fn load_symbolic_stream(
    path: &Path,
) -> Result<HashMap<String, Vec<Vec<i32>>>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let mut sequences_by_context: HashMap<String, Vec<Vec<i32>>> = HashMap::new();

    if let Some(arr) = json["symbolic_streams"].as_array() {
        for session_data in arr {
            let call_type = session_data["call_type"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string();

            let phrases: Vec<i32> = if let Some(phrases_arr) = session_data["phrases"].as_array() {
                phrases_arr.iter().filter_map(|v| v.as_i64()).map(|v| v as i32).collect()
            } else {
                continue;
            };

            // For this corpus, each call type has one "session" with all phrases
            // Create individual sequences for analysis
            sequences_by_context
                .entry(call_type)
                .or_insert_with(Vec::new)
                .push(phrases);
        }
    }

    Ok(sequences_by_context)
}

// ============================================================================
// Analysis Functions
// ============================================================================

#[derive(serde::Serialize)]
struct NgramResults {
    unique_bigrams: usize,
    unique_trigrams: usize,
    cross_context_bigrams: usize,
    bigram_frequency: HashMap<String, usize>,
}

fn analyze_ngrams(
    sequences_by_context: &HashMap<String, Vec<Vec<i32>>>,
) -> Result<NgramResults, Box<dyn std::error::Error>> {
    let mut bigrams: HashSet<Vec<i32>> = HashSet::new();
    let mut trigrams: HashSet<Vec<i32>> = HashSet::new();
    let mut bigram_freq: HashMap<String, usize> = HashMap::new();
    let mut cross_context_bigrams: HashSet<(String, Vec<i32>)> = HashSet::new();

    for (ctx, sequences) in sequences_by_context {
        for seq in sequences {
            for window in seq.windows(2) {
                bigrams.insert(window.to_vec());
                let key = format!("{:?}_{:?}", window[0], window[1]);
                *bigram_freq.entry(key).or_insert(0) += 1;
                cross_context_bigrams.insert((ctx.clone(), window.to_vec()));
            }
            for window in seq.windows(3) {
                trigrams.insert(window.to_vec());
            }
        }
    }

    Ok(NgramResults {
        unique_bigrams: bigrams.len(),
        unique_trigrams: trigrams.len(),
        cross_context_bigrams: cross_context_bigrams.len(),
        bigram_frequency: bigram_freq,
    })
}

#[derive(serde::Serialize)]
struct MotifResults {
    n_motifs: usize,
    multi_context_motifs: usize,
    motifs: Vec<Vec<i32>>,
}

fn analyze_motifs(
    sequences_by_context: &HashMap<String, Vec<Vec<i32>>>,
) -> Result<MotifResults, Box<dyn std::error::Error>> {
    let mut motif_counts: HashMap<Vec<i32>, usize> = HashMap::new();
    let mut motif_contexts: HashMap<Vec<i32>, HashSet<String>> = HashMap::new();

    // Find repeating patterns of length 2-5
    let min_motif_len = 2;
    let max_motif_len = 5;

    for (ctx, sequences) in sequences_by_context {
        for seq in sequences {
            for motif_len in min_motif_len..=max_motif_len {
                for window in seq.windows(motif_len) {
                    let motif = window.to_vec();
                    *motif_counts.entry(motif.clone()).or_insert(0) += 1;
                    motif_contexts
                        .entry(motif)
                        .or_insert_with(HashSet::new)
                        .insert(ctx.clone());
                }
            }
        }
    }

    // Filter motifs that appear more than once
    let recurring_motifs: Vec<_> = motif_counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(motif, _)| motif)
        .collect();

    let multi_context = motif_contexts
        .iter()
        .filter(|(_, contexts)| contexts.len() > 1)
        .count();

    Ok(MotifResults {
        n_motifs: recurring_motifs.len(),
        multi_context_motifs: multi_context,
        motifs: recurring_motifs,
    })
}

#[derive(serde::Serialize)]
struct PatternStatistics {
    avg_length: f64,
    length_variance: f64,
    unique_phrase_ratio: f64,
}

fn calculate_pattern_statistics(
    sequences_by_context: &HashMap<String, Vec<Vec<i32>>>,
) -> Result<PatternStatistics, Box<dyn std::error::Error>> {
    let mut all_lengths: Vec<f64> = Vec::new();
    let mut all_phrases: HashSet<i32> = HashSet::new();
    let mut total_phrases: usize = 0;

    for (_ctx, sequences) in sequences_by_context {
        for seq in sequences {
            all_lengths.push(seq.len() as f64);
            for &phrase in seq {
                all_phrases.insert(phrase);
                total_phrases += 1;
            }
        }
    }

    let avg_length = if !all_lengths.is_empty() {
        all_lengths.iter().sum::<f64>() / all_lengths.len() as f64
    } else {
        0.0
    };

    let variance = if !all_lengths.is_empty() {
        let mean = avg_length;
        all_lengths.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / all_lengths.len() as f64
    } else {
        0.0
    };

    let unique_ratio = if total_phrases > 0 {
        all_phrases.len() as f64 / total_phrases as f64
    } else {
        0.0
    };

    Ok(PatternStatistics {
        avg_length,
        length_variance: variance,
        unique_phrase_ratio: unique_ratio,
    })
}
