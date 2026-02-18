// Create Phrase-Level Marmoset Corpus from Frame-Level Corpus
//
// Usage: cargo run --release --example create_marmoset_phrase_level_corpus

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║    Creating Phrase-Level Marmoset Corpus                                ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  This converts frame-level corpus to phrase-level for analysis               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let start_time = std::time::Instant::now();

    // Configuration
    let frame_corpus_path = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_corpus_for_analysis.json");
    let phrase_corpus_path = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phrase_level_corpus.json");

    // ========================================================================
    // Load Frame-Level Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Frame-Level Corpus                                     │");
    println!("└───────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let content = std::fs::read_to_string(&frame_corpus_path)?;
    let json: serde_json::from_str(&content)?;

    let cluster_to_phrase: &HashMap<String, String> = json["cluster_to_phrase"]
        .as_object()
        .map(|obj| obj.iter().filter_map(|(k, v)| {
            v.as_str().map(|s| (k.clone(), s.to_string()))
        })).collect()
        .unwrap_or_default();

    let metadata = json["metadata"].as_object().map(|(k, v)| (k.as_str().unwrap_or("").to_string(), v.as_str().unwrap_or("").to_string())).collect();

    println!("   📂 Loaded frame-level corpus");
    println!("      • Sessions: {}", metadata["num_sessions"]);
    println!("      • Total phrases: {}", metadata["total_phrases"]);
    println!("      • Vocabulary size: {}", metadata["vocabulary_size"]);
    println!();

    // ========================================================================
    // Build Phrase-Level Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Phrase-Level Corpus                                   │");
    println!("└───────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut sessions: Vec<serde_json::Value> = Vec::new();

    for (i, session_data) in json["sessions"].as_array()
        .ok_or("Sessions not found").enumerate() {
        let call_type = "Vocalization";

        let phrases: Vec<i32> = if let Some(arr) = session_data.as_array() {
            arr.iter().filter_map(|v| v.as_i64()).map(|v| v as i32).collect()
        } else {
            continue;
        };

        sessions.push(serde_json::json!({
            "session_id": i,
            "call_type": call_type,
            "phrases": phrases,
        }));
    }

    let metadata_out = serde_json::json!({
        "description": "Phrase-level marmoset corpus (segmented from frame-level)",
        "species": "marmoset",
        "num_sessions": sessions.len(),
        "total_phrases": sessions.iter().map(|s| s["phrases"].as_array().map(|a| a.len()).unwrap_or(0)).sum::<usize>(),
        "vocabulary_size": cluster_to_phrase.len(),
    });

    // ========================================================================
    // Save Phrase-Level Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Saving Phrase-Level Corpus                                   │");
    println!("└───────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    let output = serde_json::to_string_pretty(&serde_json::json!({
        "metadata": metadata_out,
        "sessions": sessions,
        "cluster_to_phrase": cluster_to_phrase,
        "phrase_to_cluster": cluster_to_phrase.iter().map(|(k, v)| v).collect(),
    })?;

    std::fs::write(&phrase_corpus_path, output)?;

    println!("   💾 Saved phrase-level corpus to {}", phrase_corpus_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    CORPUS CONVERSION COMPLETE                           ║");
    println!("╠═════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 SUMMARY:                                                         ║");
    println!("║     • Frame-level corpus: {}", frame_corpus_path.display());
    println!("║     • Phrase-level corpus: {}", phrase_corpus_path.display());
    println!("║     • Total sessions: {}", sessions.len());
    println!("║     • Total phrases: {}", cluster_to_phrase.len());
    println!("║                                                                           ║");
    println!("║  ⏱️  Conversion time: {:.2}s                                       ║", elapsed.as_secs_f64());
    println!("║   📁 Results saved to:                                              ║");
    println!("║     {}                                              ║", phrase_corpus_path.display());
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}
