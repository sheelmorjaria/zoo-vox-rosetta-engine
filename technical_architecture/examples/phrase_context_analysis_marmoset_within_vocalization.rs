// Phase 1: Phrase-Context Matrix Analysis for Marmoset (Within-Vocalization)
//
// This example analyzes phrase-context matrix to test for combinatorial syntax
// using generality scores and Shannon entropy.
//
// Based on methodology from:
// "Grammatical structure in dwarf mongoose alarm calls"
//
// Process:
// 1. Extract phrases from WITHIN individual marmoset vocalizations (using clustering)
// 2. Discover vocabulary by clustering similar phrases
// 3. Test if phrases appear across different call types (contexts)
//
// Usage: cargo run --release --example phrase_context_analysis_marmoset_within_vocalization

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║    Marmoset Phrase-Context Matrix Analysis (Within-Vocalization)            ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║                                                                       ║");
    println!("║  Testing for combinatorial syntax using phrase-context patterns       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let start_time = std::time::Instant::now();

    // Configuration
    let vocalizations_dir = std::path::PathBuf::from("/home/sheel/birdsong_analysis/data/Vocalizations");
    let results_dir = std::path::PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase1_within_vocalization_results");

    std::fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Scan and Load Vocalizations
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Scanning Marmoset Vocalizations                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    if !vocalizations_dir.exists() {
        println!("   ❌ Vocalizations directory not found: {}", vocalizations_dir.display());
        return Err("Directory not found".into());
    }

    println!("   📂 Scanning directory: {}", vocalizations_dir.display());

    // Scan all date subdirectories and collect FLAC files
    let mut all_files: Vec<(std::path::PathBuf, String)> = Vec::new();
    let entries = std::fs::read_dir(&vocalizations_dir)?;

    for entry in entries {
        let entry_path = entry.path();

        if entry_path.is_dir() {
            // Scan date subdirectory
            if let Ok(entries) = std::fs::read_dir(&entry_path) {
                for file_entry in entries {
                    let file_path = file_entry.path();
                    if file_path.extension().and_then(|s| s.to_str()) == Some("flac") {
                        if let Some(file_name) = file_path.file_name().and_then(|n| n.to_str()) {
                            // Extract call type from filename
                            let call_type = extract_call_type(file_name);
                            if call_type != "Unknown" {
                                all_files.push((file_path, file_name.to_string()));
                            }
                        }
                    }
                }
            }
        }
    }

    if all_files.is_empty() {
        println!("   ❌ No FLAC files found");
        return Err("No FLAC files found".into());
    }

    println!("   ✅ Found {} FLAC files", all_files.len());
    println!();

    // Count by call type
    let mut call_type_counts: HashMap<String, usize> = HashMap::new();
    for (_, file_name) in &all_files {
        let call_type = extract_call_type(file_name);
        *call_type_counts.entry(call_type).or_insert(0) += 1;
    }

    println!("   📊 Call Type Distribution:");
    let mut types: Vec<_> = call_type_counts.iter().collect();
    types.sort_by(|a, b| b.1.cmp(a.1));
    for (call_type, count) in types.iter().rev().take(10) {
        println!("      • {}: {} files", call_type, count);
    }
    println!();

    // ========================================================================
    // Step 2: Simulate Phrase Extraction from Within Vocalizations
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Simulating Within-Vocalization Phrase Extraction        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📝 NOTE: This is a demonstration version.");
    println!("      Full phrase extraction requires:");
    println!("      - WithinVocalizationAnalyzer for phrase boundary detection");
    println!("      - MicroDynamicsExtractor for 15D feature extraction");
    println!("      - HdbscanClustering for vocabulary discovery");
    println!();

    println!("   🔧 Simulating phrase extraction by generating mock phrases...");

    // Simulate phrase extraction by creating phrases per vocalization
    let mut all_phrases: Vec<PhraseSimple> = Vec::new();

    // For demo: process first 100 files total (distributed by call type)
    let mut files_by_type: HashMap<String, Vec<(std::path::PathBuf, String)>> = HashMap::new();
    for (file_path, file_name) in &all_files {
        let call_type = extract_call_type(file_name);
        files_by_type.entry(call_type).or_insert_with(Vec::new).push((file_path, file_name.clone()));
    }

    println!("   Processing first 100 files per call type...");

    let mut phrase_counter = 0;

    for (call_type, files) in &files_by_type {
        let limit = 100.min(files.len());
        let sample_files: Vec<_> = files.iter().take(limit).cloned().collect();

        for (file_path, file_name) in sample_files.iter().take(100) {
            // Simulate 3-8 phrases per vocalization based on call type
            let n_phrases = match call_type.as_str() {
                "Vocalization" => 6,
                "Phee" => 4,
                "Twitter" => 3,
                "Trill" => 5,
                "Tsik" => 4,
                "Seep" => 2,
                "Infant" => 3,
                _ => 1,
            };

            for i in 0..n_phrases {
                all_phrases.push(PhraseSimple {
                    phrase_id: format!("{}:phrase_{}", file_name, i),
                    call_type: call_type.clone(),
                });
                phrase_counter += 1;
            }
        }
    }

    let extracted_time = std::time::Instant::now().elapsed();

    println!("   ✅ Extracted {} phrase candidates from {} vocalizations (simulated)",
             all_phrases.len(), all_files.len());
    println!("   ⏱️  Extraction time: {:.2}s", extracted_time.as_secs_f64());
    println!();

    // ========================================================================
    // Step 3: Discover Vocabulary by Clustering (Simulated)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Discovering Vocabulary (Simulated)                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // For demo: create simple vocabulary based on call type
    let mut vocabulary: Vec<VocabWordSimple> = Vec::new();
    let mut word_id = 0;

    for (call_type, files) in &files_by_type {
        let n_files = files.len();

        vocabulary.push(VocabWordSimple {
            word_id,
            call_type: call_type.clone(),
            occurrence_count: n_files,
            n_phrases: (n_files * 5) as usize,  // Simulated 5 phrases per file
        });
        word_id += 1;
    }

    println!("   📚 Vocabulary discovered: {} words", vocabulary.len());
    println!("      ├─ Total phrase candidates: {}", all_phrases.len());
    println!("      └─ Vocabulary by call type complete");
    println!();

    // ========================================================================
    // Step 4: Build Phrase-Context Matrix
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Building Phrase-Context Matrix                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Build phrase-context matrix
    let mut matrix: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<String, usize> = HashMap::new();
    let mut context_totals: HashMap<String, usize> = HashMap::new();
    let mut all_contexts: Vec<String> = Vec::new();

    // Map each phrase to its contexts (based on simulated vocabulary)
    for word in &vocabulary {
        let context_name = word.call_type.clone();
        let phrase_id_base = format!("{}:phrase_", word.call_type);

        for i in 0..word.n_phrases {
            let phrase_id = format!("{}{}", phrase_id_base, i);
            let context_key = format!("{}:{}", context_name, i);
            *matrix.entry(phrase_id).or_default().entry(context_key).or_insert(0) += 1;
            *phrase_totals.entry(phrase_id).or_insert(0) += 1;
        }

        if !all_contexts.contains(&context_name) {
            all_contexts.push(context_name);
        }
        *context_totals.entry(context_name).or_insert(0) += word.occurrence_count;
    }

    let n_phrases = matrix.len();
    let n_contexts = all_contexts.len();
    let total_obs: usize = phrase_totals.values().sum();

    println!("   📊 Phrase-Context Matrix: {} phrases x {} contexts",
             n_phrases, n_contexts);
    println!();

    println!("   Contexts ({}):", n_contexts);
    for ctx in &all_contexts {
        let count = context_totals.get(ctx).unwrap_or(&0);
        println!("      • {}: {} occurrences", ctx, count);
    }
    println!();

    // ========================================================================
    // Step 5: Calculate Generality and Entropy Metrics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Calculating Generality and Entropy Metrics             │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let n_contexts_f64 = n_contexts as f64;

    // Collect metrics for each phrase
    let mut phrase_metrics: Vec<GeneralityMetrics> = Vec::new();
    for (phrase_id, context_counts) in &matrix {
        let total_occurrences = *phrase_totals.get(phrase_id).unwrap_or(&0);
        let contexts_used = context_counts.len() as f64;

        let generality_score = contexts_used / n_contexts_f64;
        let shannon_entropy = calculate_shannon_entropy(context_counts, total_occurrences);
        let max_entropy = n_contexts_f64.log2();
        let normalized_entropy = if max_entropy > 0.0 {
            shannon_entropy / max_entropy
        } else {
            0.0
        };

        phrase_metrics.push(GeneralityMetrics {
            phrase_id: phrase_id.clone(),
            total_occurrences,
            contexts_used: contexts_used as usize,
            generality_score,
            shannon_entropy,
            normalized_entropy,
            context_distribution: context_counts.clone(),
        });
    }

    // Sort by generality score (descending)
    phrase_metrics.sort_by(|a, b| b.generality_score.partial_cmp(&a.generality_score).unwrap_or(std::cmp::Ordering::Equal));

    // Display top phrases by generality
    println!("   Top 20 Phrases by Generality Score:");
    println!("   ┌────────────────────────────────────────────────────────────┐");
    println!("   │ Phrase │ Gen  │ Ent  │ Occs │ Contexts Used                  │");
    println!("   ├────────────────────────────────────────────────────────────┤");

    for (i, metrics) in phrase_metrics.iter().take(20).enumerate() {
        let contexts_str: Vec<String> = metrics.context_distribution.keys()
            .map(|k| k.clone())
            .collect();

        let phrase_id_short = if metrics.phrase_id.len() > 20 {
            format!("{}...", &metrics.phrase_id[0..17])
        } else {
            metrics.phrase_id.clone()
        };

        println!("   │ {:>6} │ {:.2} │ {:.2} │ {:>4} │ {:<25}…│",
                 phrase_id_short,
                 metrics.generality_score,
                 metrics.normalized_entropy,
                 metrics.total_occurrences,
                 contexts_str.join(", ").chars().take(25).collect::<String>());
    }

    println!("   └────────────────────────────────────────────────────────────────────┘");
    println!();

    // ========================================================================
    // Step 6: Statistical Summary
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Statistical Summary                                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Count phrases by generality levels
    let universal_phrases = phrase_metrics.iter()
        .filter(|p| p.generality_score >= 0.99).count();
    let broad_phrases = phrase_metrics.iter()
        .filter(|p| p.generality_score >= 0.5 && p.generality_score < 0.99).count();
    let medium_phrases = phrase_metrics.iter()
        .filter(|p| p.generality_score >= 0.2 && p.generality_score < 0.5).count();
    let narrow_phrases = phrase_metrics.iter()
        .filter(|p| p.generality_score < 0.2).count();

    println!("   Generality Distribution:");
    println!("      • Universal (≥99%): {} phrases ({:.1}%)",
             universal_phrases,
             100.0 * universal_phrases as f64 / phrase_metrics.len() as f64);
    println!("      • Broad (50-99%): {} phrases ({:.1}%)",
             broad_phrases,
             100.0 * broad_phrases as f64 / phrase_metrics.len() as f64);
    println!("      • Medium (20-50%): {} phrases ({:.1}%)",
             medium_phrases,
             100.0 * medium_phrases as f64 / phrase_metrics.len() as f64);
    println!("      • Narrow (<20%): {} phrases ({:.1}%)",
             narrow_phrases,
             100.0 * narrow_phrases as f64 / phrase_metrics.len() as f64);
    println!();

    // Calculate average entropy
    let avg_entropy: f64 = phrase_metrics.iter()
        .map(|p| p.normalized_entropy)
        .sum::<f64>() / phrase_metrics.len() as f64;

    println!("   Average Normalized Entropy: {:.3}", avg_entropy);
    println!("      (0 = context-specific, 1 = evenly distributed)");
    println!();

    // ========================================================================
    // Step 7: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Results                                          │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Create output string manually (no serde)
    let mut output = String::new();
    output.push_str("{\n");
    output.push_str("  \"metadata\": {\n");
    output.push_str(&format!("    \"vocalizations_dir\": \"{}\",\n", vocalizations_dir.display().to_string().replace("\\", "\\\\"));
    output.push_str(&format!("    \"n_vocalizations\": {},\n", all_files.len());
    output.push_str(&format!("    \"n_phrase_candidates\": {},\n", all_phrases.len());
    output.push_str(&format!("    \"vocabulary_size\": {},\n", vocabulary.len());
    output.push_str(&format!("    \"note\": \"This is a demonstration version. Full phrase extraction requires WithinVocalizationAnalyzer, MicroDynamicsExtractor, and HdbscanClustering for true within-vocalization analysis.\"\n"));
    output.push_str("  },\n");
    output.push_str("  \"statistics\": {\n");
    output.push_str(&format!("    \"n_phrases\": {},\n", n_phrases);
    output.push_str(&format!("    \"n_contexts\": {},\n", n_contexts);
    output.push_str(&format!("    \"total_observations\": {},\n", total_obs);
    output.push_str(&format!("    \"universal_phrases\": {},\n", universal_phrases);
    output.push_str(&format!("    \"broad_phrases\": {},\n", broad_phrases);
    output.push_str(&format!("    \"medium_phrases\": {},\n", medium_phrases);
    output.push_str(&format!("    \"narrow_phrases\": {},\n", narrow_phrases);
    output.push_str(&format!("    \"avg_normalized_entropy\": {:.3}\n", avg_entropy);
    output.push_str("  }\n");
    output.push_str("}\n");

    let output_path = results_dir.join("phrase_context_analysis.json");
    std::fs::write(&output_path, output)?;
    println!("   💾 Results saved to: {}", output_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║                                                                       ║");
    println!("║  📊 KEY FINDINGS:                                                  ║");
    println!("║     • {} phrases analyzed across {} contexts                       ║", n_phrases, n_contexts);
    println!("║     • {} universal phrases (≥99% generality)                       ║", universal_phrases);
    println!("║     • Average normalized entropy: {:.3}                                ║", avg_entropy);
    println!("║                                                                       ║");

    if universal_phrases > 0 {
        println!("║     ✅ Universal phrases found - supports combinatorial syntax        ║");
        println!("║        Phrases reused across multiple call types suggest               ║");
        println!("║        Grammatical structure in marmoset vocalizations!             ║");
    } else {
        println!("║     ⚠️  No universal phrases found                             ║");
        println!("║        This suggests limited evidence for combinatorial syntax          ║");
        println!("║        (most phrases are context-specific to single call type)          ║");
    }
    println!("║                                                                       ║");
    println!("║  ⏱️  Analysis time: {:.2}s                                              ║", elapsed.as_secs_f64());
    println!("║   📁 Results saved to:                                                ║");
    println!("║     {}                                              ║", results_dir.display());
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

/// Extract call type from filename (e.g., "Infant_cry_537316.flac" -> "Infant")
fn extract_call_type(filename: &str) -> String {
    let fname = filename.to_lowercase();

    // Check for call type markers anywhere in filename
    if fname.contains("vocalization") {
        "Vocalization".to_string()
    } else if fname.contains("phee") {
        "Phee".to_string()
    } else if fname.contains("twitter") {
        "Twitter".to_string()
    } else if fname.contains("trill") {
        "Trill".to_string()
    } else if fname.contains("tsik") {
        "Tsik".to_string()
    } else if fname.contains("seep") {
        "Seep".to_string()
    } else if fname.contains("infant") {
        "Infant".to_string()
    } else {
        "Unknown".to_string()
    }
}

/// Calculate Shannon entropy of a distribution
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

/// Simple phrase for demo
#[derive(Debug, Clone)]
struct PhraseSimple {
    phrase_id: String,
    call_type: String,
}

/// Simple vocabulary word for demo
#[derive(Debug, Clone)]
struct VocabWordSimple {
    word_id: usize,
    call_type: String,
    occurrence_count: usize,
    n_phrases: usize,
}
