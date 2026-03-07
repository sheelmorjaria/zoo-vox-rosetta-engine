// Minimal Marmoset Phrase-Context Analysis Example
//
// Usage: cargo run --release --example phrase_context_analysis_marmoset_simple_minimal

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║    Marmoset Phrase-Context Matrix Analysis                                 ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Analyzing marmoset corpus for phrase-context patterns...            ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();

    let start_time = Instant::now();

    // Configuration
    let corpus_path = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_corpus_for_analysis.json");
    let results_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase1_generality_results");

    fs::create_dir_all(&results_dir)?;

    // Step 1: Load Corpus
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Corpus                                                │");
    println!("└───────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let content = fs::read_to_string(&corpus_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let sessions_array = json["sessions"].as_array().ok_or("Sessions not found")?;

    println!("   📂 Loaded {} sessions", sessions_array.len());
    println!();

    // Step 2: Build Phrase-Context Matrix
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Phrase-Context Matrix                              │");
    println!("└───────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut matrix: HashMap<i32, HashMap<String, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<i32, usize> = HashMap::new();
    let mut context_totals: HashMap<String, usize> = HashMap::new();

    // Build matrix from sessions - limit to first 1000 for speed
    for session_data in sessions_array.iter().take(1000) {
        let context = "Vocalization".to_string();

        if let Some(arr) = session_data.as_array() {
            for phrase_id in arr.iter().filter_map(|v| v.as_i64()).map(|v| v as i32) {
                if phrase_id >= 0 {
                    *matrix.entry(phrase_id).or_default().entry(context.clone()).or_insert(0) += 1;
                    *phrase_totals.entry(phrase_id).or_insert(0) += 1;
                    *context_totals.entry(context.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    let n_phrases = matrix.len();
    let n_contexts = context_totals.len();
    let total_obs: usize = phrase_totals.values().sum::<usize>();

    println!("   📊 Matrix: {} phrases x {} contexts", n_phrases, n_contexts);
    println!();

    // Step 3: Calculate Metrics
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Calculating Metrics                                           │");
    println!("└───────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let n_contexts_f64 = context_totals.len() as f64;

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

        println!(
            "      Phrase {}: gen={:.2}, ent={:.2}",
            phrase_id, generality_score, normalized_entropy
        );
    }

    let elapsed = start_time.elapsed();

    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                   ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 Results:                                                         ║");
    println!(
        "║     • Phrases analyzed: {}                                        ║",
        n_phrases
    );
    println!(
        "║     • Contexts: {}                                               ║",
        n_contexts
    );
    println!(
        "║     • Total observations: {}                                          ║",
        total_obs
    );
    println!(
        "║     • Analysis time: {:.2}s                                        ║",
        elapsed.as_secs_f64()
    );
    println!("║   📁 Results saved to:                                                ║");
    println!(
        "║     {}                                              ║",
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
