// Phase 1: Phrase-Context Matrix Analysis for Marmoset
//
// Analyzes marmoset vocalization corpus to test for combinatorial syntax
// Uses the corpus JSON file directly without complex audio processing
//
// Usage: cargo run --release --example phrase_context_analysis_marmoset_simple

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;
use serde::{Serialize, Deserialize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║    Phase 1: Phrase-Context Matrix Analysis - Marmoset                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  HYPOTHESIS TESTING: Combinatorial Syntax vs Holistic Signals             ║");
    println!("║                                                                           ║");
    println!("║  If combinatorial syntax exists:                                          ║");
    println!("║    • General-purpose phrases (used in many contexts) - function words     ║");
    println!("║    • Context-specific phrases (used in few contexts) - content words       ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let corpus_path = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_corpus_for_analysis.json");
    let results_dir = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_phase1_generality_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Marmoset Corpus                                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let content = fs::read_to_string(corpus_path)?;
    let json: serde_json::from_str(&content)?;

    // Get basic info
    let cluster_to_phrase: &HashMap<String, String> = json["cluster_to_phrase"]
        .as_object()
        .map(|obj| obj.iter().filter_map(|(k, v)| {
            v.as_str().map(|s| (k.clone(), s.to_string()))
        })).collect())
        .unwrap_or_default();

    let metadata = CorpusMetadata {
        description: json["metadata"]["description"].as_str().unwrap_or("").to_string(),
        num_sessions: json["metadata"]["num_sessions"].as_u64().unwrap_or(0) as usize,
        species: json["metadata"]["species"].as_str().unwrap_or("marmoset").to_string(),
        total_phrases: json["metadata"]["total_phrases"].as_u64().unwrap_or(0) as usize,
        vocabulary_size: json["metadata"]["vocabulary_size"].as_u64().unwrap_or(0) as usize,
    };

    let sessions_array = json["sessions"].as_array().ok_or("Sessions not found")?;

    println!("   📂 Loaded {} sessions", sessions_array.len());
    println!("   📊 Metadata:");
    println!("      • Species: {}", metadata.species);
    println!("      • Sessions: {}", metadata.num_sessions);
    println!("      • Total phrases: {}", metadata.total_phrases);
    println!("      • Vocabulary size: {}", metadata.vocabulary_size);
    println!();

    // ========================================================================
    // Step 2: Build Phrase-Context Matrix
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Phrase-Context Matrix                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut matrix: HashMap<i32, HashMap<String, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<i32, usize> = HashMap::new();
    let mut context_totals: HashMap<String, usize> = HashMap::new();

    // Build matrix from sessions
    // Default context to "Vocalization" for all phrases since corpus doesn't have call type info
    for session_data in sessions_array.iter().take(1000) {
        let context = "Vocalization".to_string();

        if let Some(arr) = session_data.as_array() {
            for phrase_id in arr.iter().filter_map(|v| v.as_i64()).map(|v| v as i32) {
                if phrase_id < 0 {
                    *matrix.entry(phrase_id).or_default().entry(context.clone()).or_insert(0) += 1;
                    *phrase_totals.entry(phrase_id).or_insert(0) += 1;
                    *context_totals.entry(context).or_insert(0) += 1;
                }
            }
        }
    }

    let n_phrases = matrix.len();
    let n_contexts = context_totals.len();
    let total_obs: usize = phrase_totals.values().sum::<usize>();

    println!("   📊 Matrix Statistics:");
    println!("      ├─ Unique phrases: {}", n_phrases);
    println!("      ├─ Behavioral contexts: {}", n_contexts);
    println!("      └─ Total observations: {}", total_obs);
    println!();

    // ========================================================================
    // Step 3: Calculate Generality Metrics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Calculating Generality Metrics                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📈 Computing generality metrics for {} phrases...", n_phrases);
    println!();

    let mut metrics: Vec<GeneralityMetrics> = Vec::new();

    for (&phrase_id, context_counts) in &matrix {
        let total_occurrences = phrase_totals[phrase_id];
        let contexts_used = context_counts.len();

        let generality_score = if n_contexts > 0 {
            contexts_used as f64 / n_contexts as f64
        } else {
            0.0
        };

        let shannon_entropy = calculate_shannon_entropy(context_counts, total_occurrences);
        let max_entropy = (n_contexts as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 {
            shannon_entropy / max_entropy
        } else {
            0.0
        };

        metrics.push(GeneralityMetrics {
            phrase_id,
            total_occurrences,
            contexts_used,
            total_contexts: n_contexts,
            generality_score,
            shannon_entropy,
            normalized_entropy,
            classification: PhraseType::Rare,
        });
    }

    println!("   ✅ Computed {} metrics", metrics.len());
    println!();

    // ========================================================================
    // Step 4: Classify Phrase Types
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Phrase Type Classification                                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    classify_phrases(&mut metrics);

    let type_counts = count_phrase_types(&metrics);
    println!("   📊 Phrase Type Distribution:");
    println!("      ┌────────────────────────────┬──────────┬──────────┐");
    println!("      │ Type                       │ Count    │ Percentage│");
    println!("      ├────────────────────────────┼──────────┼──────────┤");
    println!("      │ Universal Generalist       │ {:8} │ {:8.1}│",
             type_counts.0, type_counts.0 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Generalist                 │ {:8} │ {:8.1}│",
             type_counts.1, type_counts.1 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Flexible Specialist        │ {:8} │ {:8.1}│",
             type_counts.2, type_counts.2 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Context Specialist         │ {:8} │ {:8.1}│",
             type_counts.3, type_counts.3 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Highly Specific            │ {:8} │ {:8.1}│",
             type_counts.4, type_counts.4 as f64 / metrics.len() as f64 * 100.0);
    println!("      └────────────────────────────┴──────────┴──────────┘");
    println!();

    // ========================================================================
    // Step 5: Permutation Test
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Permutation Test (Statistical Significance)                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   🎲 Running permutation test...");
    println!();

    let n_permutations = 1000;
    let perm_result = run_permutation_test(&matrix, &phrase_totals, n_permutations, n_contexts)?;

    println!();
    println!("   ✅ Permutation Test Results:");
    println!("      ├─ Observed mean generality: {:.4}", perm_result.observed_mean_generality);
    println!("      ├─ Null mean generality:      {:.4} ± {:.4}",
             perm_result.null_mean_generality, perm_result.null_std_generality);
    println!("      ├─ Z-score:                   {:.4}", perm_result.z_score);
    println!("      ├─ P-value:                   {:.6}", perm_result.p_value);
    println!("      └─ Significant (α=0.05):      {}", if perm_result.significant { "YES ✨" } else { "NO" });
    println!();

    if perm_result.significant {
        println!("   🎯 CONCLUSION: Observed phrase reuse is significantly NON-RANDOM.");
        println!("      This supports hypothesis of intentional combinatorial syntax!");
    } else {
        println!("   ⚠️  CONCLUSION: Observed pattern could be due to random chance.");
        println!("      More evidence needed to support combinatorial syntax.");
    }
    println!();

    // ========================================================================
    // Step 6: Summary Statistics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Summary Statistics                                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let summary = compute_summary_statistics(&metrics);
    println!("   📊 Generality Score Distribution:");
    println!("      ├─ Mean:   {:.4}", summary.mean_generality_score);
    println!("      ├─ Median: {:.4}", summary.median_generality_score);
    println!("      └─ Std:    {:.4}", summary.std_generality_score);
    println!();

    println!("   📊 Shannon Entropy Distribution:");
    println!("      ├─ Mean:   {:.4} bits", summary.mean_shannon_entropy);
    println!("      └─ Median: {:.4} bits", summary.median_shannon_entropy);
    println!();

    // ========================================================================
    // Step 7: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let results = AnalysisResults {
        metadata: MetaData {
            dataset: "marmoset".to_string(),
            n_phrases,
            n_contexts,
            total_observations: total_obs,
            n_sessions: 1000,
            analysis_timestamp: chrono::Utc::now().to_rfc3339(),
        },
        phrase_context_matrix: serialize_pcm(&PhraseContextMatrix {
            matrix,
            phrase_totals,
            context_totals,
        }),
        generality_metrics: metrics.clone(),
        permutation_test: perm_result.clone(),
        summary_statistics: summary.clone(),
    };

    let results_path = results_dir.join("generality_analysis_results.json");
    fs::write(&results_path, serde_json::to_string_pretty(&results)?)?;
    println!("   💾 Full results: {}", results_path.display());
    println!();

    let csv_path = results_dir.join("phrase_generality_metrics.csv");
    save_generality_csv(&metrics, &csv_path, cluster_to_phrase)?;
    println!("   💾 Generality CSV: {}", csv_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                     ║");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 KEY FINDINGS:                                                         ║");
    println!("║     • Total phrases analyzed: {}                                        ║", n_phrases);
    println!("║     • Call type contexts: {}                                            ║", n_contexts);
    println!("║     • Generalist phrases: {} ({:.1}%)                                   ║",
             type_counts.0 + type_counts.1,
             (type_counts.0 + type_counts.1) as f64 / metrics.len() as f64 * 100.0);
    println!("║     • Specialist phrases: {} ({:.1}%)                                   ║",
             type_counts.3 + type_counts.4,
             (type_counts.3 + type_counts.4) as f64 / metrics.len() as f64 * 100.0);
    println!("║                                                                           ║");
    println!("║  🧪 STATISTICAL TEST:                                                     ║");
    if perm_result.significant {
        println!("║     ✅ SIGNIFICANT: Phrase reuse is non-random (p={:.4})                ║", perm_result.p_value);
        println!("║     This SUPPORTS combinatorial syntax hypothesis                   ║");
    } else {
        println!("║     ⚠️  NOT SIGNIFICANT                                                 ║");
        println!("║     Insufficient evidence for combinatorial syntax                      ║");
    }
    println!("║                                                                           ║");
    println!("║  ⏱️  Analysis time: {:.2}s                                                ║", elapsed.as_secs_f64());
    println!("║                                                                           ║");
    println!("║  📁 Results saved to:                                                     ║");
    println!("║     {}                                              ║", results_dir.display());
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, serde_json::Serialize, serde_json::Deserialize)]
struct CorpusMetadata {
    description: String,
    num_sessions: usize,
    species: String,
    total_phrases: usize,
    vocabulary_size: usize,
}

#[derive(Debug, Clone, serde_json::Serialize)]
struct MetaData {
    dataset: String,
    n_phrases: usize,
    n_contexts: usize,
    total_observations: usize,
    n_sessions: usize,
    analysis_timestamp: String,
}

#[derive(Debug, Clone, serde_json::Serialize)]
struct PhraseContextMatrix {
    matrix: HashMap<i32, HashMap<String, usize>>,
    phrase_totals: HashMap<i32, usize>,
    context_totals: HashMap<String, usize>,
}

#[derive(Debug, Clone, serde_json::Serialize)]
struct GeneralityMetrics {
    phrase_id: i32,
    total_occurrences: usize,
    contexts_used: usize,
    total_contexts: usize,
    generality_score: f64,
    shannon_entropy: f64,
    normalized_entropy: f64,
    classification: PhraseType,
}

#[derive(Debug, Clone, Copy, PartialEq, serde_json::Serialize)]
enum PhraseType {
    UniversalGeneralist,  // Used in all contexts
    Generalist,           // Used in most contexts
    FlexibleSpecialist,  // Used in several contexts
    ContextSpecialist,   // Used in few contexts
    HighlySpecific,       // Used almost exclusively in one context
    Rare,                 // Very low frequency
}

#[derive(Debug, Clone, serde_json::Serialize)]
struct AnalysisResults {
    metadata: MetaData,
    phrase_context_matrix: PhraseContextMatrix,
    generality_metrics: Vec<GeneralityMetrics>,
    permutation_test: PermutationTestResult,
    summary_statistics: SummaryStatistics,
}

#[derive(Debug, Clone, serde_json::Serialize)]
struct PermutationTestResult {
    observed_mean_generality: f64,
    null_mean_generality: f64,
    null_std_generality: f64,
    p_value: f64,
    z_score: f64,
    significant: bool,
    n_permutations: usize,
}

#[derive(Debug, Clone, serde_json::Serialize)]
struct SummaryStatistics {
    n_universal_phrases: usize,
    n_generalist_phrases: usize,
    n_flexible_specialist_phrases: usize,
    n_context_specialist_phrases: usize,
    n_highly_specific_phrases: usize,
    mean_generality_score: f64,
    median_generality_score: f64,
    std_generality_score: f64,
    mean_shannon_entropy: f64,
    median_shannon_entropy: f64,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn calculate_shannon_entropy(context_counts: &HashMap<String, usize>, total: usize) -> f64 {
    let mut entropy = 0.0f64;
    for &count in context_counts.values() {
        if count > 0 && total > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn classify_phrases(metrics: &mut [GeneralityMetrics]) {
    for m in metrics.iter_mut() {
        m.classification = if m.total_occurrences < 5 {
            PhraseType::Rare
        } else if m.generality_score >= 0.8 {
            PhraseType::UniversalGeneralist
        } else if m.generality_score >= 0.5 {
            PhraseType::Generalist
        } else if m.normalized_entropy >= 0.6 {
            PhraseType::FlexibleSpecialist
        } else if m.generality_score >= 0.2 {
            PhraseType::ContextSpecialist
        } else {
            PhraseType::HighlySpecific
        };
    }
}

fn count_phrase_types(metrics: &[GeneralityMetrics]) -> (usize, usize, usize, usize, usize) {
    let mut counts = (0, 0, 0, 0, 0);
    for m in metrics {
        match m.classification {
            PhraseType::UniversalGeneralist => counts.0 += 1,
            PhraseType::Generalist => counts.1 += 1,
            PhraseType::FlexibleSpecialist => counts.2 += 1,
            PhraseType::ContextSpecialist => counts.3 += 1,
            PhraseType::HighlySpecific => counts.4 += 1,
            PhraseType::Rare => {}
        }
    }
    counts
}

fn run_permutation_test(
    matrix: &HashMap<i32, HashMap<String, usize>>,
    phrase_totals: &HashMap<i32, usize>,
    n_permutations: usize,
    n_contexts: usize,
) -> PermutationTestResult {
    use rand::Rng;

    // Calculate observed mean generality
    let observed_gens: Vec<f64> = matrix.keys()
        .map(|phrase_id| {
            context_counts.len() as f64 / n_contexts as f64
        })
        .collect();

    let observed_mean = observed_gens.iter().sum::<f64>() / observed_gens.len() as f64;

    // Collect all (context, phrase_id) pairs
    let mut all_pairs: Vec<(String, i32)> = Vec::new();
    for (phrase_id, context_counts) in &matrix {
        for (context, &count) in context_counts.iter()filter(|(ctx, _)| ctx.0 == 1) {
            for _ in 0..*count {
                all_pairs.push((context.clone(), phrase_id));
            }
        }
    }

    let total_pairs = all_pairs.len();

    // Run permutations
    let null_means: Vec<f64> = (0..n_permutations)
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut shuffled_contexts: Vec<String> = all_pairs.iter().map(|(ctx, _)| ctx.clone()).collect();

            // Shuffle contexts
            for i in 1..shuffled_contexts.len() {
                let j = rng.gen_range(0..i + 1);
                shuffled_contexts.swap(i, j);
            }

            let mut phrase_context_counts: HashMap<i32, usize> = HashMap::new();
            for ((_, phrase_id), _) in all_pairs.iter().zip(shuffled_contexts.iter()) {
                *phrase_context_counts.entry(*phrase_id).or_insert(0) += 1;
            }

            let gen_scores: Vec<f64> = phrase_context_counts.values()
                .map(|&n_ctx| *n_ctx as f64 / n_contexts as f64)
                .collect();

            gen_scores.iter().sum::<f64>() / gen_scores.len() as f64
        })
        .collect();

    let null_mean = null_means.iter().sum::<f64>() / null_means.len() as f64;
    let null_variance = null_means.iter()
        .map(|&x| (x - null_mean).powi(2))
        .sum::<f64>() / null_means.len() as f64;
    let null_std = null_variance.sqrt();

    // Calculate z-score and p-value
    let z_score = if null_std > 0.0 {
        (observed_mean - null_mean) / null_std
    } else {
        0.0
    };

    let count_ge_observed = null_means.iter().filter(|&&x| x >= observed_mean).count();
    let p_value = (count_ge_observed + 1) as f64 / (n_permutations + 1) as f64;

    PermutationTestResult {
        observed_mean_generality: observed_mean,
        null_mean_generality: null_mean,
        null_std_generality: null_std,
        p_value,
        z_score,
        significant: p_value < 0.05,
        n_permutations,
    }
}

fn compute_summary_statistics(metrics: &[GeneralityMetrics]) -> SummaryStatistics {
    let type_counts = count_phrase_types(metrics);

    let gen_scores: Vec<f64> = metrics.iter().map(|m| m.generality_score).collect();
    let mean_gen = gen_scores.iter().sum::<f64>() / gen_scores.len() as f64;
    let mut sorted_gen = gen_scores.clone();
    sorted_gen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_gen = sorted_gen[sorted_gen.len() / 2];
    let var_gen = gen_scores.iter().map(|x| (x - mean_gen).powi(2)).sum::<f64>() / gen_scores.len() as f64;

    let entropies: Vec<f64> = metrics.iter().map(|m| m.shannon_entropy).collect();
    let mean_ent = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let mut sorted_ent = entropies.clone();
    sorted_ent.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ent = sorted_ent[sorted_ent.len() / 2];

    SummaryStatistics {
        n_universal_phrases: type_counts.0,
        n_generalist_phrases: type_counts.1,
        n_flexible_specialist_phrases: type_counts.2,
        n_context_specialist_phrases: type_counts.3,
        n_highly_specific_phrases: type_counts.4,
        mean_generality_score: mean_gen,
        median_generality_score: median_gen,
        std_generality_score: var_gen.sqrt(),
        mean_shannon_entropy: mean_ent,
        median_shannon_entropy: median_ent,
    }
}

fn serialize_pcm(pcm: &PhraseContextMatrix) -> PhraseContextMatrix {
    pcm.clone()
}

fn save_generality_csv(
    metrics: &[GeneralityMetrics],
    path: &Path,
    cluster_to_phrase: &HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut wtr = csv::WriterBuilder::new().from_path(path)?;

    wtr.write_record(&[
        "phrase_id",
        "total_occurrences",
        "contexts_used",
        "generality_score",
        "shannon_entropy",
        "normalized_entropy",
        "classification",
        "phrase_description",
    ])?;

    for m in metrics {
        let class_str = match m.classification {
            PhraseType::UniversalGeneralist => "Universal Generalist",
            PhraseType::Generalist => "Generalist",
            PhraseType::FlexibleSpecialist => "Flexible Specialist",
            PhraseType::ContextSpecialist => "Context Specialist",
            PhraseType::HighlySpecific => "Highly Specific",
            PhraseType::Rare => "Rare",
        };

        let phrase_desc = cluster_to_phrase.get(&m.phrase_id.to_string())
            .map(|s| s.clone())
            .unwrap_or_else(|| format!("phrase_{}", m.phrase_id));

        wtr.write_record(&[
            m.phrase_id.to_string(),
            m.total_occurrences.to_string(),
            m.contexts_used.to_string(),
            format!("{:.4}", m.generality_score),
            format!("{:.4}", m.shannon_entropy),
            format!("{:.4}", m.normalized_entropy),
            class_str.to_string(),
            phrase_desc,
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
