// Phase 1: Phrase-Context Matrix Analysis for Marmoset
//
// This example analyzes the phrase-context matrix to test for combinatorial syntax
// using generality scores and Shannon entropy.
//
// Based on methodology from:
// "Grammatical structure in dwarf mongoose alarm calls"
// - generality score: contexts used / total contexts
// - Shannon entropy: distribution evenness across contexts
//
// Usage: cargo run --release --example phrase_context_analysis_marmoset

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║    Marmoset Phrase-Context Matrix Analysis                               ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                       ║");
    println!("║  Testing for combinatorial syntax using phrase-context patterns       ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();

    let start_time = Instant::now();

    // Configuration
    let corpus_path =
        PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phrase_level_corpus.json");
    let results_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase1_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Phrase-Level Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Phrase-Level Corpus                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let content = fs::read_to_string(&corpus_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let metadata = json["metadata"].as_object().ok_or("Metadata not found")?;
    let sessions_array = json["sessions"].as_array().ok_or("Sessions not found")?;

    let total_phrases: usize = metadata["total_phrases"].as_u64().unwrap_or(0) as usize;
    let vocabulary_size: usize = metadata["vocabulary_size"].as_u64().unwrap_or(0) as usize;

    println!("   📂 Loaded phrase-level corpus");
    println!("      • Total phrases: {}", total_phrases);
    println!("      • Vocabulary size: {}", vocabulary_size);
    println!("      • Sessions: {}", sessions_array.len());
    println!();

    // ========================================================================
    // Step 2: Build Phrase-Context Matrix
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Phrase-Context Matrix                             │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Build the phrase-context matrix
    // Rows: phrase IDs (0 to vocabulary_size - 1)
    // Columns: call types (Vocalization, Twitter, Tsik, Phee, Trill, Infant_cry, Seep)
    let mut matrix: HashMap<i32, HashMap<String, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<i32, usize> = HashMap::new();
    let mut context_totals: HashMap<String, usize> = HashMap::new();
    let mut all_contexts: Vec<String> = Vec::new();

    for session_data in sessions_array {
        let call_type = session_data["call_type"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string();

        if !all_contexts.contains(&call_type) {
            all_contexts.push(call_type.clone());
        }

        if let Some(arr) = session_data["phrases"].as_array() {
            for phrase_value in arr {
                if let Some(phrase_id) = phrase_value.as_i64() {
                    let phrase_id = phrase_id as i32;
                    *matrix
                        .entry(phrase_id)
                        .or_default()
                        .entry(call_type.clone())
                        .or_insert(0) += 1;
                    *phrase_totals.entry(phrase_id).or_insert(0) += 1;
                    *context_totals.entry(call_type.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    let n_contexts = all_contexts.len();
    let total_observations: usize = phrase_totals.values().sum();

    println!(
        "   📊 Phrase-Context Matrix: {} phrases x {} contexts",
        matrix.len(),
        n_contexts
    );
    println!();
    println!("   Contexts ({}):", n_contexts);
    for ctx in &all_contexts {
        let count = context_totals.get(ctx).unwrap_or(&0);
        println!("      • {}: {} occurrences", ctx, count);
    }
    println!();

    // ========================================================================
    // Step 3: Calculate Generality and Entropy Metrics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Calculating Metrics                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let n_contexts_f64 = n_contexts as f64;

    // Collect metrics for each phrase
    let mut phrase_metrics: Vec<PhraseMetrics> = Vec::new();
    for (&phrase_id, context_counts) in &matrix {
        let total_occurrences = phrase_totals[&phrase_id];
        let contexts_used = context_counts.len() as f64;

        let generality_score = contexts_used / n_contexts_f64;
        let shannon_entropy = calculate_shannon_entropy(context_counts, total_occurrences);
        let max_entropy = n_contexts_f64.log2();
        let normalized_entropy = if max_entropy > 0.0 {
            shannon_entropy / max_entropy
        } else {
            0.0
        };

        phrase_metrics.push(PhraseMetrics {
            phrase_id,
            generality_score,
            shannon_entropy,
            normalized_entropy,
            total_occurrences,
            contexts_used: contexts_used as usize,
            context_distribution: context_counts.clone(),
        });
    }

    // Sort by generality score (descending)
    phrase_metrics.sort_by(|a, b| {
        b.generality_score
            .partial_cmp(&a.generality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Display top phrases by generality
    println!("   Top 20 Phrases by Generality Score:");
    println!("   ┌────────────────────────────────────────────────────────────────────┐");
    println!("   │ Phrase │ Gen  │ Ent  │ Occs │ Contexts Used               │");
    println!("   ├────────────────────────────────────────────────────────────────────┤");

    for (i, metrics) in phrase_metrics.iter().take(20).enumerate() {
        let contexts_str: Vec<String> = metrics
            .context_distribution
            .keys()
            .map(|k| k.clone())
            .collect();

        println!(
            "   │ {:6} │ {:.2} │ {:.2} │ {:5} │ {:25}…│",
            metrics.phrase_id,
            metrics.generality_score,
            metrics.normalized_entropy,
            metrics.total_occurrences,
            contexts_str.join(", ").chars().take(25).collect::<String>()
        );
    }

    println!("   └────────────────────────────────────────────────────────────────────┘");
    println!();

    // ========================================================================
    // Step 4: Statistical Summary
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Statistical Summary                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Count phrases by generality levels
    let universal_phrases = phrase_metrics
        .iter()
        .filter(|p| p.generality_score >= 0.99)
        .count();
    let broad_phrases = phrase_metrics
        .iter()
        .filter(|p| p.generality_score >= 0.5 && p.generality_score < 0.99)
        .count();
    let medium_phrases = phrase_metrics
        .iter()
        .filter(|p| p.generality_score >= 0.2 && p.generality_score < 0.5)
        .count();
    let narrow_phrases = phrase_metrics
        .iter()
        .filter(|p| p.generality_score < 0.2)
        .count();

    println!("   Generality Distribution:");
    println!(
        "      • Universal (≥99%): {} phrases ({:.1}%)",
        universal_phrases,
        100.0 * universal_phrases as f64 / phrase_metrics.len() as f64
    );
    println!(
        "      • Broad (50-99%): {} phrases ({:.1}%)",
        broad_phrases,
        100.0 * broad_phrases as f64 / phrase_metrics.len() as f64
    );
    println!(
        "      • Medium (20-50%): {} phrases ({:.1}%)",
        medium_phrases,
        100.0 * medium_phrases as f64 / phrase_metrics.len() as f64
    );
    println!(
        "      • Narrow (<20%): {} phrases ({:.1}%)",
        narrow_phrases,
        100.0 * narrow_phrases as f64 / phrase_metrics.len() as f64
    );
    println!();

    // Calculate average entropy
    let avg_entropy: f64 = phrase_metrics
        .iter()
        .map(|p| p.normalized_entropy)
        .sum::<f64>()
        / phrase_metrics.len() as f64;

    println!("   Average Normalized Entropy: {:.3}", avg_entropy);
    println!("      (0 = context-specific, 1 = evenly distributed)");
    println!();

    // ========================================================================
    // Step 5: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Saving Results                                               │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save detailed metrics
    let results = serde_json::json!({
        "metadata": {
            "corpus_path": corpus_path.display().to_string(),
            "n_phrases": matrix.len(),
            "n_contexts": n_contexts,
            "total_observations": total_observations,
            "vocabulary_size": vocabulary_size,
            "contexts": all_contexts,
        },
        "statistics": {
            "universal_phrases": universal_phrases,
            "broad_phrases": broad_phrases,
            "medium_phrases": medium_phrases,
            "narrow_phrases": narrow_phrases,
            "avg_normalized_entropy": avg_entropy,
        },
        "phrase_metrics": phrase_metrics,
    });

    let output_path = results_dir.join("phrase_context_analysis.json");
    fs::write(&output_path, serde_json::to_string_pretty(&results)?)?;
    println!("   💾 Results saved to: {}", output_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE 1 ANALYSIS COMPLETE                           ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║                                                                       ║");
    println!("║  📊 KEY FINDINGS:                                                     ║");
    println!(
        "║     • {} phrases analyzed across {} contexts                       ║",
        matrix.len(),
        n_contexts
    );
    println!(
        "║     • {} universal phrases (≥99% generality)                       ║",
        universal_phrases
    );
    println!(
        "║     • Average normalized entropy: {:.3}                                ║",
        avg_entropy
    );
    println!("║                                                                       ║");
    if universal_phrases > 0 {
        println!("║     ✅ Universal phrases found - supports combinatorial syntax        ║");
    } else {
        println!("║     ⚠️  No universal phrases - limited evidence for combinatorial        ║");
        println!("║        syntax                                                          ║");
    }
    println!("║                                                                       ║");
    println!(
        "║  ⏱️  Analysis time: {:.2}s                                              ║",
        elapsed.as_secs_f64()
    );
    println!(
        "║   📁 Results: {}                               ║",
        results_dir.display()
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

fn calculate_shannon_entropy(context_counts: &HashMap<String, usize>, total: usize) -> f64 {
    let mut entropy = 0.0f64;
    for count in context_counts.values() {
        if *count > 0 && total > 0 {
            let p = *count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

#[derive(serde::Serialize)]
struct PhraseMetrics {
    phrase_id: i32,
    generality_score: f64,
    shannon_entropy: f64,
    normalized_entropy: f64,
    total_occurrences: usize,
    contexts_used: usize,
    context_distribution: HashMap<String, usize>,
}
