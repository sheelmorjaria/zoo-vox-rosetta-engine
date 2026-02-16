// Phrase-Context Analysis for Egyptian Fruit Bat
//
// This analysis tests whether bat vocalizations exhibit combinatorial syntax
// by analyzing the distribution of phrases (vocabulary items) across behavioral contexts.
//
// Hypothesis: If combinatorial syntax exists, we should find:
// 1. General-purpose phrases appearing in multiple contexts (high entropy)
// 2. Context-specific phrases appearing in few contexts (low entropy)
//
// Usage: cargo run --release --example phrase_context_analysis_bat

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let annotations_path = data_dir.join("annotations.csv");
    let results_dir = data_dir.join("lexicon_to_syntax_results");
    let clusters_path = results_dir.join("minibatch_clusters.json");
    let output_path = results_dir.join("phrase_context_analysis.json");

    println!("🦇 Phrase-Context Analysis: Egyptian Fruit Bat");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Testing: Combinatorial Syntax Hypothesis");
    println!();

    // ========================================================================
    // Step 1: Load cluster labels
    // ========================================================================

    println!("📂 Step 1: Loading cluster labels...");
    println!();

    let clusters_json = std::fs::read_to_string(&clusters_path)?;
    let clusters_data: serde_json::Value = serde_json::from_str(&clusters_json)?;

    let labels: Vec<i32> = clusters_data["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();

    println!("   └─ Loaded {} phrase labels", labels.len());
    println!();

    // ========================================================================
    // Step 2: Load annotations to get contexts
    // ========================================================================

    println!("📂 Step 2: Loading behavioral context annotations...");
    println!();

    // Create map from file name to context
    let mut file_to_context: HashMap<String, String> = HashMap::new();

    if let Ok(file) = File::open(&annotations_path) {
        let reader = BufReader::new(file);

        // Skip header
        for line in reader.lines().skip(1) {
            if let Ok(line) = line {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let file_name = format!("{}.wav", parts[0]); // Add .wav extension

                    // Extract context from parts[1] (behavioral context)
                    let context = if parts.len() >= 3 && !parts[2].is_empty() {
                        parts[2].to_string()
                    } else {
                        "General".to_string()
                    };

                    file_to_context.insert(file_name, context);
                }
            }
        }

        println!("   └─ Loaded {} file-context mappings", file_to_context.len());
    } else {
        println!("   ⚠️  Annotations file not found, using synthetic contexts");
    }
    println!();

    // ========================================================================
    // Step 3: Build phrase-context matrix
    // ========================================================================

    println!("🔄 Step 3: Building phrase-context matrix...");
    println!();

    // Get list of all unique contexts
    let mut contexts_set: HashSet<String> = file_to_context.values().cloned().collect();
    let mut contexts: Vec<String> = contexts_set.into_iter().collect();
    contexts.sort(); // Sort for consistent ordering

    let n_contexts = contexts.len();

    // Map each file (phrase) to its context
    // Note: The labels array corresponds to processed files in alphabetical order
    // We need to match them correctly

    let mut phrase_context_matrix: HashMap<i32, HashMap<String, usize>> = HashMap::new();
    let audio_dir = data_dir.join("audio");

    // Get list of processed files (sorted alphabetically)
    let mut processed_files: Vec<String> = std::fs::read_dir(&audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .filter_map(|entry| {
            entry.file_name()
                .to_str()
                .map(|s| s.to_string())
        })
        .collect();
    processed_files.sort();

    // Limit to the number of labels we have
    let n_processed = labels.len().min(processed_files.len());

    println!("   Processing {} file-context mappings...", n_processed);

    for i in 0..n_processed {
        let label = labels[i];
        let file_name = &processed_files[i];

        // Get context for this file
        let context = file_to_context.get(file_name)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string());

        // Update phrase-context matrix
        *phrase_context_matrix
            .entry(label)
            .or_insert_with(HashMap::new)
            .entry(context.clone())
            .or_insert(0) += 1;
    }

    let n_clusters = phrase_context_matrix.len();

    println!("   └─ Created {} x {} matrix", n_clusters, n_contexts);
    println!();

    // ========================================================================
    // Step 4: Calculate generality scores and entropy
    // ========================================================================

    println!("📈 Step 4: Calculating Generality Scores and Entropy...");
    println!();

    let mut phrase_analysis: Vec<PhraseAnalysis> = Vec::new();

    for (&cluster_id, context_counts) in &phrase_context_matrix {
        let total_count: usize = context_counts.values().sum();
        let n_contexts_used = context_counts.len();

        // Generality Score: proportion of contexts used
        let generality_score = n_contexts_used as f64 / n_contexts.max(1) as f64;

        // Shannon Entropy: measures uniformity of distribution
        let mut entropy = 0.0;
        for count in context_counts.values() {
            if *count > 0 {
                let p = *count as f64 / total_count as f64;
                entropy -= p * p.log2();
            }
        }

        // Max entropy for this number of contexts
        let max_entropy = if n_contexts_used > 0 {
            (n_contexts_used as f64).log2()
        } else {
            0.0
        };

        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Determine classification
        let classification = if generality_score >= 0.8 {
            "General-Purpose"
        } else if generality_score >= 0.4 {
            "Multi-Context"
        } else {
            "Context-Specific"
        };

        phrase_analysis.push(PhraseAnalysis {
            phrase_id: cluster_id,
            total_count,
            n_contexts_used,
            generality_score,
            entropy,
            normalized_entropy,
            classification: classification.to_string(),
        });
    }

    // Sort by generality score
    phrase_analysis.sort_by(|a, b| b.generality_score.partial_cmp(&a.generality_score).unwrap());

    println!("   └─ Calculated metrics for {} phrases", phrase_analysis.len());
    println!();

    // ========================================================================
    // Step 5: Display results
    // ========================================================================

    println!("📊 Step 5: Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("Contexts found ({} total):", n_contexts);
    for (i, ctx) in contexts.iter().enumerate() {
        println!("   {}. {}", i + 1, ctx);
    }
    println!();

    println!("General-Purpose Phrases (appear in most/all contexts):");
    println!("─────────────────────────────────────────────────");
    let general_purpose: Vec<_> = phrase_analysis.iter()
        .filter(|p| p.classification == "General-Purpose")
        .collect();

    for phrase in general_purpose.iter().take(10) {
        println!("   Phrase {:2}: Generality={:.2}, Entropy={:.2}, Count={}",
            phrase.phrase_id, phrase.generality_score, phrase.entropy, phrase.total_count);
    }
    println!("   Total: {} general-purpose phrases", general_purpose.len());
    println!();

    println!("Multi-Context Phrases (moderate reusability):");
    println!("──────────────────────────────────────────────");
    let multi_context: Vec<_> = phrase_analysis.iter()
        .filter(|p| p.classification == "Multi-Context")
        .collect();

    for phrase in multi_context.iter().take(10) {
        println!("   Phrase {:2}: Generality={:.2}, Entropy={:.2}, Count={}",
            phrase.phrase_id, phrase.generality_score, phrase.entropy, phrase.total_count);
    }
    println!("   Total: {} multi-context phrases", multi_context.len());
    println!();

    println!("Context-Specific Phrases (highly specialized):");
    println!("──────────────────────────────────────────────");
    let context_specific: Vec<_> = phrase_analysis.iter()
        .filter(|p| p.classification == "Context-Specific")
        .collect();

    for phrase in context_specific.iter().take(10) {
        println!("   Phrase {:2}: Generality={:.2}, Entropy={:.2}, Count={}",
            phrase.phrase_id, phrase.generality_score, phrase.entropy, phrase.total_count);
    }
    println!("   Total: {} context-specific phrases", context_specific.len());
    println!();

    // ========================================================================
    // Step 6: Statistical Summary
    // ========================================================================

    println!("📈 Step 6: Statistical Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let n_general = general_purpose.len();
    let n_multi = multi_context.len();
    let n_specific = context_specific.len();
    let total = phrase_analysis.len();

    println!("   Distribution:");
    println!("   ├─ General-Purpose: {} ({:.1}%)", n_general, n_general as f64 / total as f64 * 100.0);
    println!("   ├─ Multi-Context: {} ({:.1}%)", n_multi, n_multi as f64 / total as f64 * 100.0);
    println!("   └─ Context-Specific: {} ({:.1}%)", n_specific, n_specific as f64 / total as f64 * 100.0);
    println!();

    // Average entropy by classification
    let avg_entropy_general = if n_general > 0 {
        general_purpose.iter()
            .map(|p| p.normalized_entropy)
            .sum::<f64>() / n_general as f64
    } else {
        0.0
    };

    let avg_entropy_multi = if n_multi > 0 {
        multi_context.iter()
            .map(|p| p.normalized_entropy)
            .sum::<f64>() / n_multi as f64
    } else {
        0.0
    };

    let avg_entropy_specific = if n_specific > 0 {
        context_specific.iter()
            .map(|p| p.normalized_entropy)
            .sum::<f64>() / n_specific as f64
    } else {
        0.0
    };

    println!("   Average Normalized Entropy:");
    println!("   ├─ General-Purpose: {:.3}", avg_entropy_general);
    println!("   ├─ Multi-Context: {:.3}", avg_entropy_multi);
    println!("   └─ Context-Specific: {:.3}", avg_entropy_specific);
    println!();

    // ========================================================================
    // Step 7: Hypothesis Testing
    // ========================================================================

    println!("🔬 Step 7: Hypothesis Testing: Combinatorial Syntax");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let has_general_purpose = n_general > 0;
    let has_context_specific = n_specific > 0;
    let mixture_ratio = (n_general + n_multi) as f64 / total as f64;

    println!("   Prediction: Combinatorial syntax requires BOTH:");
    println!("   1. General-purpose phrases (structure/function words)");
    println!("   2. Context-specific phrases (content/meaning words)");
    println!();

    println!("   Results:");
    println!("   ├─ General-purpose phrases found: {}", has_general_purpose);
    println!("   ├─ Context-specific phrases found: {}", has_context_specific);
    println!("   └─ Mixture ratio: {:.1}%", mixture_ratio * 100.0);
    println!();

    if has_general_purpose && has_context_specific {
        println!("   ✅ SUPPORTS HYPOTHESIS: Both general and specific phrases found");
        println!("      → Evidence for combinatorial sentence structure");
        println!("      → General phrases can serve as structural frames");
        println!("      → Specific phrases can provide content/meaning");
    } else if !has_general_purpose && has_context_specific {
        println!("   ❌ REFUTES HYPOTHESIS: Only context-specific phrases found");
        println!("      → Suggests holistic, reflexive vocalizations");
        println!("      → No evidence for reusable building blocks");
    } else if has_general_purpose && !has_context_specific {
        println!("   ⚠️  INCONCLUSIVE: Only general-purpose phrases found");
        println!("      → May indicate limited behavioral contexts sampled");
        println!("      → Or oversimplified clustering");
    } else {
        println!("   ⚠️  INCONCLUSIVE: All phrases are multi-context");
        println!("      → Suggests continuous vocal space");
        println!("      → May need different clustering resolution");
    }
    println!();

    // ========================================================================
    // Step 8: Save results
    // ========================================================================

    println!("💾 Step 8: Saving results...");
    println!();

    let output_json = serde_json::json!({
        "n_phrases": total,
        "n_contexts": n_contexts,
        "contexts": contexts,
        "general_purpose_count": n_general,
        "multi_context_count": n_multi,
        "context_specific_count": n_specific,
        "mixture_ratio": mixture_ratio,
        "hypothesis_supported": has_general_purpose && has_context_specific,
        "phrase_analysis": phrase_analysis,
    });

    std::fs::write(&output_path, output_json.to_string())?;
    println!("   └─ Saved to {}", output_path.display());
    println!();

    println!("✅ Analysis Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, serde::Serialize)]
struct PhraseAnalysis {
    phrase_id: i32,
    total_count: usize,
    n_contexts_used: usize,
    generality_score: f64,
    entropy: f64,
    normalized_entropy: f64,
    classification: String,
}
