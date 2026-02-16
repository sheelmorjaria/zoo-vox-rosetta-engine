// Phase 1: Phrase-Context Matrix Analysis for Egyptian Fruit Bat
//
// This analysis tests the "Sentence Structure" hypothesis by measuring lexical flexibility.
//
// Hypothesis: If bats use combinatorial syntax, we should observe:
// 1. General-purpose phrases (used in many contexts) - "function words"
// 2. Context-specific phrases (used in few contexts) - "content words"
//
// Methods:
// - Generality Score: Contexts containing phrase / Total contexts
// - Shannon Entropy: Distribution evenness across contexts
// - Permutation Test: Statistical significance vs random chance
//
// Output: Phrase-Context matrix, generality scores, statistical tests, visualizations
//
// Usage: cargo run --release --example phrase_context_analysis_bat_generality

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;
use rand::Rng;
use rayon::prelude::*;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Annotation {
    emitter: i32,
    addressee: i32,
    context: i32,
    #[serde(rename = "Emitter pre-vocalization action")]
    emitter_pre_action: i32,
    #[serde(rename = "Addressee pre-vocalization action")]
    addressee_pre_action: i32,
    #[serde(rename = "Emitter post-vocalization action")]
    emitter_post_action: i32,
    #[serde(rename = "Addressee post-vocalization action")]
    addressee_post_action: i32,
    #[serde(rename = "File Name")]
    file_name: String,
}

#[derive(Debug, Clone)]
struct PhraseContextMapping {
    phrase_id: i32,
    context: i32,
    file_name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GeneralityMetrics {
    phrase_id: i32,
    total_occurrences: usize,
    contexts_used: usize,
    total_contexts: usize,
    generality_score: f64,  // 0.0 (context-specific) to 1.0 (universal)
    shannon_entropy: f64,    // 0.0 (specialized) to max (uniform)
    normalized_entropy: f64, // 0.0 to 1.0
    intra_context_cv: f64,   // Coefficient of variation within contexts
    classification: PhraseType,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum PhraseType {
    UniversalGeneralist,  // Used in all contexts (like "the", "and")
    Generalist,           // Used in most contexts
    FlexibleSpecialist,   // Used in several contexts with bias
    ContextSpecialist,    // Used primarily in one context
    HighlySpecific,       // Used almost exclusively in one context
    Rare,                 // Very low frequency
}

#[derive(Debug, Clone, serde::Serialize)]
struct PermutationTestResult {
    observed_mean_generality: f64,
    null_mean_generality: f64,
    null_std_generality: f64,
    p_value: f64,
    z_score: f64,
    significant: bool,
    n_permutations: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnalysisResults {
    metadata: Metadata,
    phrase_context_matrix: PhraseContextMatrix,
    generality_metrics: Vec<GeneralityMetrics>,
    permutation_test: PermutationTestResult,
    summary_statistics: SummaryStatistics,
    visualizations: VisualizationData,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Metadata {
    dataset: String,
    n_phrases: usize,
    n_contexts: usize,
    total_observations: usize,
    analysis_timestamp: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PhraseContextMatrix {
    matrix: HashMap<i32, HashMap<i32, usize>>, // phrase_id -> context_id -> count
    phrase_totals: HashMap<i32, usize>,
    context_totals: HashMap<i32, usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
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

#[derive(Debug, Clone, serde::Serialize)]
struct VisualizationData {
    phrase_type_distribution: Vec<(String, usize)>,
    generality_distribution: Vec<f64>,
    entropy_distribution: Vec<f64>,
    context_overlap_matrix: Vec<Vec<i32>>,
    upset_plot_data: UpSetPlotData,
}

#[derive(Debug, Clone, serde::Serialize)]
struct UpSetPlotData {
    context_names: Vec<String>,
    phrase_sets: Vec<PhraseSetIntersection>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PhraseSetIntersection {
    contexts: Vec<i32>,           // Which contexts are in this intersection
    phrase_count: usize,          // Number of phrases in this intersection
    example_phrases: Vec<i32>,    // Example phrase IDs
}

// ============================================================================
// Main Function
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║    Phase 1: Phrase-Context Matrix Analysis - Egyptian Fruit Bat          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  HYPOTHESIS TESTING: Combinatorial Syntax vs Holistic Signals             ║");
    println!("║                                                                           ║");
    println!("║  If combinatorial syntax exists:                                          ║");
    println!("║    • General-purpose phrases (used in many contexts) - function words     ║");
    println!("║    • Context-specific phrases (used in few contexts) - content words       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let results_dir = data_dir.join("phase1_generality_analysis_results");
    let phase0_results_dir = data_dir.join("phase0_symbolic_stream_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Data
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Data                                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Load annotations
    println!("   📂 Loading annotations...");
    let annotations = load_annotations(data_dir.join("annotations.csv"))?;
    println!("      └─ Loaded {} annotations", annotations.len());
    println!();

    // Load symbolic stream from Phase 0
    println!("   📂 Loading symbolic stream from Phase 0...");
    let (phrase_labels, file_names) = load_symbolic_stream(&phase0_results_dir)?;
    println!("      └─ Loaded {} phrase labels", phrase_labels.len());
    println!();

    // ========================================================================
    // Step 2: Build Phrase-Context Matrix
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Phrase-Context Matrix                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let pcm = build_phrase_context_matrix(&annotations, &file_names, &phrase_labels)?;

    let n_phrases = pcm.matrix.len();
    let n_contexts = pcm.context_totals.len();
    let total_obs: usize = pcm.phrase_totals.values().sum();

    println!("   📊 Matrix Statistics:");
    println!("      ├─ Unique phrases: {}", n_phrases);
    println!("      ├─ Behavioral contexts: {}", n_contexts);
    println!("      └─ Total observations: {}", total_obs);
    println!();

    // Display context distribution
    println!("   📋 Context Distribution:");
    let mut context_vec: Vec<_> = pcm.context_totals.iter().collect();
    context_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (i, (ctx, count)) in context_vec.iter().enumerate().take(10) {
        println!("      Context {:2}: {} observations ({:.1}%)",
                 ctx, count, **count as f64 / total_obs as f64 * 100.0);
    }
    println!();

    // ========================================================================
    // Step 3: Calculate Generality Metrics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Calculating Generality Metrics                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📈 Computing:");
    println!("      ├─ Generality Score (contexts used / total contexts)");
    println!("      ├─ Shannon Entropy (distribution evenness)");
    println!("      ├─ Normalized Entropy (0.0 to 1.0)");
    println!("      └─ Coefficient of Variation (within-context consistency)");
    println!();

    let mut metrics = calculate_generality_metrics(&pcm, n_contexts)?;
    println!("      └─ Computed metrics for {} phrases", metrics.len());
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
    println!("   🏷️  Phrase Type Distribution:");
    println!("      ┌────────────────────────────┬──────────┬──────────┐");
    println!("      │ Type                       │ Count    │ Percentage│");
    println!("      ├────────────────────────────┼──────────┼──────────┤");
    println!("      │ Universal Generalist       │ {:8} │ {:8.1}│", type_counts.0, type_counts.0 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Generalist                 │ {:8} │ {:8.1}│", type_counts.1, type_counts.1 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Flexible Specialist        │ {:8} │ {:8.1}│", type_counts.2, type_counts.2 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Context Specialist         │ {:8} │ {:8.1}│", type_counts.3, type_counts.3 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Highly Specific            │ {:8} │ {:8.1}│", type_counts.4, type_counts.4 as f64 / metrics.len() as f64 * 100.0);
    println!("      └────────────────────────────┴──────────┴──────────┘");
    println!();

    // Display top phrases of each type
    display_example_phrases_by_type(&metrics, &pcm);
    println!();

    // ========================================================================
    // Step 5: Permutation Test for Statistical Significance
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Permutation Test (Statistical Significance)                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   🎲 Running permutation test to determine if observed generality...");
    println!("      is significantly different from random chance.");
    println!();

    let n_permutations = 1000;
    println!("      ├─ Number of permutations: {}", n_permutations);
    println!("      ├─ Null hypothesis: Phrase-context associations are random");
    println!("      └─ Testing...");

    let perm_result = run_permutation_test(&pcm, n_permutations, n_contexts)?;

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
        println!("      This supports the hypothesis of intentional combinatorial syntax!");
        println!();
    } else {
        println!("   ⚠️  CONCLUSION: Observed pattern could be due to random chance.");
        println!("      More evidence needed to support combinatorial syntax.");
        println!();
    }

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
    // Step 7: Generate Visualization Data
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Generating Visualization Data                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let viz_data = generate_visualization_data(&pcm, &metrics, n_contexts)?;
    println!("   ✅ Visualization data generated");
    println!();

    // ========================================================================
    // Step 8: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 8: Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let results = AnalysisResults {
        metadata: Metadata {
            dataset: "egyptian_fruit_bat".to_string(),
            n_phrases,
            n_contexts,
            total_observations: total_obs,
            analysis_timestamp: chrono::Utc::now().to_rfc3339(),
        },
        phrase_context_matrix: serialize_pcm(&pcm),
        generality_metrics: metrics.clone(),
        permutation_test: perm_result.clone(),
        summary_statistics: summary.clone(),
        visualizations: viz_data,
    };

    // Save full results
    let results_path = results_dir.join("generality_analysis_results.json");
    fs::write(&results_path, serde_json::to_string_pretty(&results)?)?;
    println!("   💾 Full results: {}", results_path.display());

    // Save generality metrics CSV
    let csv_path = results_dir.join("phrase_generality_metrics.csv");
    save_generality_csv(&metrics, &csv_path)?;
    println!("   💾 Generality CSV: {}", csv_path.display());

    // Save phrase-context matrix CSV
    let matrix_path = results_dir.join("phrase_context_matrix.csv");
    save_matrix_csv(&pcm, &matrix_path)?;
    println!("   💾 Matrix CSV: {}", matrix_path.display());

    // Save visualization-specific files
    let upset_path = results_dir.join("upset_plot_data.json");
    fs::write(&upset_path, serde_json::to_string_pretty(&results.visualizations.upset_plot_data)?)?;
    println!("   💾 UpSet plot data: {}", upset_path.display());

    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 KEY FINDINGS:                                                         ║");
    println!("║     • Total phrases analyzed: {}                                        ║", n_phrases);
    println!("║     • Behavioral contexts: {}                                            ║", n_contexts);
    println!("║     • Generalist phrases (potential function words): {} ({:.1}%)        ║",
             type_counts.0 + type_counts.1,
             (type_counts.0 + type_counts.1) as f64 / metrics.len() as f64 * 100.0);
    println!("║     • Specialist phrases (potential content words): {} ({:.1}%)        ║",
             type_counts.3 + type_counts.4,
             (type_counts.3 + type_counts.4) as f64 / metrics.len() as f64 * 100.0);
    println!("║                                                                           ║");
    println!("║  🧪 STATISTICAL TEST:                                                     ║");
    if perm_result.significant {
        println!("║     ✅ SIGNIFICANT: Phrase reuse is non-random (p={:.4})                ║", perm_result.p_value);
        println!("║     This SUPPORTS the combinatorial syntax hypothesis                   ║");
    } else {
        println!("║     ⚠️  NOT SIGNIFICANT: Cannot reject null hypothesis (p={:.4})         ║", perm_result.p_value);
        println!("║     Insufficient evidence for combinatorial syntax                      ║");
    }
    println!("║                                                                           ║");
    println!("║  ⏱️  Analysis time: {:.2}s                                                ║", elapsed.as_secs_f64());
    println!("║                                                                           ║");
    println!("║  📁 Results saved to:                                                     ║");
    println!("║     {}                                              ║", results_dir.display());
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ============================================================================
// Data Loading Functions
// ============================================================================

fn load_annotations(path: impl AsRef<Path>) -> Result<Vec<Annotation>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut annotations = Vec::new();

    // Skip header
    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            annotations.push(Annotation {
                emitter: parts[0].parse().unwrap_or(0),
                addressee: parts[1].parse().unwrap_or(0),
                context: parts[2].parse().unwrap_or(0),
                emitter_pre_action: parts[3].parse().unwrap_or(0),
                addressee_pre_action: parts[4].parse().unwrap_or(0),
                emitter_post_action: parts[5].parse().unwrap_or(0),
                addressee_post_action: parts[6].parse().unwrap_or(0),
                file_name: parts[7].to_string(),
            });
        }
    }

    Ok(annotations)
}

fn load_symbolic_stream(
    results_dir: &Path,
) -> Result<(Vec<i32>, Vec<String>), Box<dyn std::error::Error>> {
    // Try to load from readable CSV first
    let csv_path = results_dir.join("symbolic_stream_readable.csv");
    if csv_path.exists() {
        let content = fs::read_to_string(&csv_path)?;
        let mut labels = Vec::new();
        let mut file_names = Vec::new();

        for line in content.lines().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                file_names.push(parts[0].to_string());
                labels.push(parts[1].parse().unwrap_or(-1));
            }
        }

        return Ok((labels, file_names));
    }

    // Otherwise load from JSON and separate files
    let clusters_path = results_dir.join("hdbscan_clusters.json");
    let clusters_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&clusters_path)?)?;

    let labels = clusters_json["clustering"]["labels"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).map(|i| i as i32).collect())
        .unwrap_or_default();

    let file_names_path = results_dir.join("bat_file_names.json");
    let file_names: Vec<String> = serde_json::from_str(&fs::read_to_string(&file_names_path)?)?;

    Ok((labels, file_names))
}

// ============================================================================
// Matrix Building
// ============================================================================

fn build_phrase_context_matrix(
    annotations: &[Annotation],
    file_names: &[String],
    phrase_labels: &[i32],
) -> Result<PhraseContextMatrix, Box<dyn std::error::Error>> {
    let mut matrix: HashMap<i32, HashMap<i32, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<i32, usize> = HashMap::new();
    let mut context_totals: HashMap<i32, usize> = HashMap::new();

    // Create file_name -> context mapping
    let mut file_context_map: HashMap<String, i32> = HashMap::new();
    for ann in annotations {
        let file_key = ann.file_name.trim().to_string();
        file_context_map.insert(file_key, ann.context);
    }

    // Build matrix
    for (file_name, &phrase_id) in file_names.iter().zip(phrase_labels.iter()) {
        if phrase_id == -1 {
            continue; // Skip noise
        }

        if let Some(&context) = file_context_map.get(file_name) {
            *matrix.entry(phrase_id).or_default().entry(context).or_insert(0) += 1;
            *phrase_totals.entry(phrase_id).or_insert(0) += 1;
            *context_totals.entry(context).or_insert(0) += 1;
        }
    }

    Ok(PhraseContextMatrix {
        matrix,
        phrase_totals,
        context_totals,
    })
}

// ============================================================================
// Generality Metrics Calculation
// ============================================================================

fn calculate_generality_metrics(
    pcm: &PhraseContextMatrix,
    n_contexts: usize,
) -> Result<Vec<GeneralityMetrics>, Box<dyn std::error::Error>> {
    let mut metrics = Vec::new();

    for (&phrase_id, context_counts) in &pcm.matrix {
        let total_occurrences = pcm.phrase_totals[&phrase_id];
        let contexts_used = context_counts.len();

        // Generality score: proportion of contexts used
        let generality_score = if n_contexts > 0 {
            contexts_used as f64 / n_contexts as f64
        } else {
            0.0
        };

        // Shannon entropy
        let shannon_entropy = calculate_shannon_entropy(context_counts, total_occurrences);

        // Normalized entropy (divide by max possible entropy)
        let max_entropy = (n_contexts as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 {
            shannon_entropy / max_entropy
        } else {
            0.0
        };

        // Coefficient of variation (within-context consistency)
        let counts: Vec<f64> = context_counts.values().map(|&v| v as f64).collect();
        let mean = counts.iter().sum::<f64>() / counts.len() as f64;
        let variance = counts.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / counts.len() as f64;
        let intra_context_cv = if mean > 0.0 {
            variance.sqrt() / mean
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
            intra_context_cv,
            classification: PhraseType::Rare, // Will be updated
        });
    }

    Ok(metrics)
}

fn calculate_shannon_entropy(context_counts: &HashMap<i32, usize>, total: usize) -> f64 {
    let mut entropy = 0.0;
    for &count in context_counts.values() {
        if count > 0 && total > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ============================================================================
// Classification
// ============================================================================

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
            PhraseType::HighlySpecific | PhraseType::Rare => counts.4 += 1,
        }
    }

    counts
}

// ============================================================================
// Display Functions
// ============================================================================

fn display_example_phrases_by_type(metrics: &[GeneralityMetrics], pcm: &PhraseContextMatrix) {
    let examples_per_type = 3;

    println!("   📚 Example Phrases by Type:");
    println!();

    for &phrase_type in &[
        PhraseType::UniversalGeneralist,
        PhraseType::Generalist,
        PhraseType::ContextSpecialist,
        PhraseType::HighlySpecific,
    ] {
        let type_name = match phrase_type {
            PhraseType::UniversalGeneralist => "Universal Generalist",
            PhraseType::Generalist => "Generalist",
            PhraseType::FlexibleSpecialist => "Flexible Specialist",
            PhraseType::ContextSpecialist => "Context Specialist",
            PhraseType::HighlySpecific => "Highly Specific",
            PhraseType::Rare => "Rare",
        };

        let mut examples: Vec<_> = metrics.iter()
            .filter(|m| m.classification == phrase_type)
            .take(examples_per_type)
            .collect();

        if !examples.is_empty() {
            println!("      {}:", type_name);
            for (i, m) in examples.iter().enumerate() {
                // Get primary context
                let mut contexts: Vec<_> = pcm.matrix.get(&m.phrase_id)
                    .map(|m| m.iter().collect())
                    .unwrap_or_default();
                contexts.sort_by(|a, b| b.1.cmp(a.1));

                let primary_ctx = contexts.first()
                    .map(|(ctx, _)| **ctx)
                    .unwrap_or(0);

                println!("         {}. Phrase {:4}: gen={:.2}, ent={:.2}, freq={}, ctx={}",
                         i + 1, m.phrase_id, m.generality_score,
                         m.normalized_entropy, m.total_occurrences, primary_ctx);
            }
            println!();
        }
    }
}

// ============================================================================
// Permutation Test
// ============================================================================

fn run_permutation_test(
    pcm: &PhraseContextMatrix,
    n_permutations: usize,
    n_contexts: usize,
) -> Result<PermutationTestResult, Box<dyn std::error::Error>> {
    use rayon::prelude::*;

    // Calculate observed mean generality
    let observed_gens: Vec<f64> = pcm.matrix.keys()
        .map(|&phrase_id| {
            pcm.matrix.get(&phrase_id)
                .map(|ctxs| ctxs.len() as f64 / n_contexts as f64)
                .unwrap_or(0.0)
        })
        .collect();

    let observed_mean = observed_gens.iter().sum::<f64>() / observed_gens.len() as f64;

    // Collect all (context, phrase_id) pairs for permutation
    let mut all_pairs: Vec<(i32, i32)> = Vec::new();
    for (&phrase_id, context_counts) in &pcm.matrix {
        for (&context, &count) in context_counts {
            for _ in 0..count {
                all_pairs.push((context, phrase_id));
            }
        }
    }

    let _total_obs = all_pairs.len();

    // Run permutations in parallel
    let null_means: Vec<f64> = (0..n_permutations)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut shuffled_contexts: Vec<i32> = all_pairs.iter().map(|(ctx, _)| *ctx).collect();
            shuffled_contexts.shuffle(&mut rng);

            // Build randomized generality scores
            let mut phrase_context_counts: HashMap<i32, usize> = HashMap::new();
            for ((_, phrase_id), _shuffled_ctx) in all_pairs.iter().zip(shuffled_contexts.iter()) {
                *phrase_context_counts.entry(*phrase_id).or_insert(0) += 1;
            }

            let gen_scores: Vec<f64> = phrase_context_counts.values()
                .map(|&n_ctx| n_ctx as f64 / n_contexts as f64)
                .collect();

            gen_scores.iter().sum::<f64>() / gen_scores.len() as f64
        })
        .collect();

    // Calculate statistics
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

    // Count how many null values are >= observed (one-tailed test)
    let count_ge_observed = null_means.iter().filter(|&&x| x >= observed_mean).count();
    let p_value = (count_ge_observed + 1) as f64 / (n_permutations + 1) as f64;

    Ok(PermutationTestResult {
        observed_mean_generality: observed_mean,
        null_mean_generality: null_mean,
        null_std_generality: null_std,
        p_value,
        z_score,
        significant: p_value < 0.05,
        n_permutations,
    })
}

// ============================================================================
// Summary Statistics
// ============================================================================

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

// ============================================================================
// Visualization Data Generation
// ============================================================================

fn generate_visualization_data(
    pcm: &PhraseContextMatrix,
    metrics: &[GeneralityMetrics],
    n_contexts: usize,
) -> Result<VisualizationData, Box<dyn std::error::Error>> {
    // Phrase type distribution
    let type_counts = count_phrase_types(metrics);
    let phrase_type_distribution = vec![
        ("Universal Generalist".to_string(), type_counts.0),
        ("Generalist".to_string(), type_counts.1),
        ("Flexible Specialist".to_string(), type_counts.2),
        ("Context Specialist".to_string(), type_counts.3),
        ("Highly Specific".to_string(), type_counts.4),
    ];

    // Generality and entropy distributions
    let generality_distribution: Vec<f64> = metrics.iter()
        .map(|m| m.generality_score)
        .collect();

    let entropy_distribution: Vec<f64> = metrics.iter()
        .map(|m| m.normalized_entropy)
        .collect();

    // Context overlap matrix (for heatmap)
    let all_contexts: Vec<i32> = {
        let mut ctxs: Vec<_> = pcm.context_totals.keys().cloned().collect();
        ctxs.sort();
        ctxs
    };

    let context_overlap_matrix = all_contexts.iter().map(|&ctx_i| {
        all_contexts.iter().map(|&ctx_j| {
            // Count phrases shared between contexts i and j
            let shared = pcm.matrix.values()
                .filter(|ctxs| ctxs.contains_key(&ctx_i) && ctxs.contains_key(&ctx_j))
                .count() as i32;
            shared
        }).collect()
    }).collect();

    // UpSet plot data
    let upset_plot_data = generate_upset_data(pcm, &all_contexts)?;

    Ok(VisualizationData {
        phrase_type_distribution,
        generality_distribution,
        entropy_distribution,
        context_overlap_matrix,
        upset_plot_data,
    })
}

fn generate_upset_data(
    pcm: &PhraseContextMatrix,
    contexts: &[i32],
) -> Result<UpSetPlotData, Box<dyn std::error::Error>> {
    let context_names: Vec<String> = contexts.iter()
        .map(|c| format!("Ctx_{}", c))
        .collect();

    // Generate all possible intersections
    let mut phrase_sets: Vec<PhraseSetIntersection> = Vec::new();

    // Get all unique context sets
    let mut context_set_map: HashMap<Vec<i32>, usize> = HashMap::new();
    let mut context_set_examples: HashMap<Vec<i32>, Vec<i32>> = HashMap::new();

    for (&phrase_id, context_counts) in &pcm.matrix {
        let mut ctxs: Vec<i32> = context_counts.keys().cloned().collect();
        ctxs.sort();
        *context_set_map.entry(ctxs.clone()).or_insert(0) += 1;
        context_set_examples.entry(ctxs).or_insert_with(Vec::new).push(phrase_id);
    }

    // Convert to intersections
    for (contexts, count) in context_set_map {
        let examples = context_set_examples.get(&contexts)
            .map(|v| v.iter().take(5).cloned().collect())
            .unwrap_or_default();

        phrase_sets.push(PhraseSetIntersection {
            contexts: contexts.clone(),
            phrase_count: count,
            example_phrases: examples,
        });
    }

    // Sort by count (descending)
    phrase_sets.sort_by(|a, b| b.phrase_count.cmp(&a.phrase_count));

    Ok(UpSetPlotData {
        context_names,
        phrase_sets,
    })
}

// ============================================================================
// Serialization Helpers
// ============================================================================

fn serialize_pcm(pcm: &PhraseContextMatrix) -> PhraseContextMatrix {
    pcm.clone()
}

fn save_generality_csv(metrics: &[GeneralityMetrics], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    wtr.write_record(&[
        "phrase_id",
        "total_occurrences",
        "contexts_used",
        "generality_score",
        "shannon_entropy",
        "normalized_entropy",
        "classification",
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

        wtr.write_record(&[
            m.phrase_id.to_string(),
            m.total_occurrences.to_string(),
            m.contexts_used.to_string(),
            format!("{:.4}", m.generality_score),
            format!("{:.4}", m.shannon_entropy),
            format!("{:.4}", m.normalized_entropy),
            class_str.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

fn save_matrix_csv(pcm: &PhraseContextMatrix, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // Get sorted contexts
    let mut contexts: Vec<_> = pcm.context_totals.keys().cloned().collect();
    contexts.sort();

    // Write header
    writeln!(file, "phrase_id,{}", contexts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","))?;

    // Write rows
    let mut phrase_ids: Vec<_> = pcm.matrix.keys().cloned().collect();
    phrase_ids.sort();

    for phrase_id in phrase_ids {
        if let Some(context_counts) = pcm.matrix.get(&phrase_id) {
            let counts: Vec<String> = contexts.iter()
                .map(|c| context_counts.get(c).map(|n| n.to_string()).unwrap_or("0".to_string()))
                .collect();
            writeln!(file, "{},{}", phrase_id, counts.join(","))?;
        }
    }

    Ok(())
}

// ============================================================================
// Random Shuffle Extension
// ============================================================================

trait Shuffle<T> {
    fn shuffle(&mut self, rng: &mut rand::rngs::ThreadRng);
}

impl<T> Shuffle<T> for [T] where T: Clone {
    fn shuffle(&mut self, rng: &mut rand::rngs::ThreadRng) {
        for i in (1..self.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            self.swap(i, j);
        }
    }
}
