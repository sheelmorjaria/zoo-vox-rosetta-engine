// Phrase-Context Matrix Analysis for Meerkat Vocalizations
//
// Tests the "Sentence Structure" hypothesis by measuring lexical flexibility.
//
// Hypothesis: If meerkats use combinatorial syntax, we should observe:
// 1. General-purpose phrases (used in many contexts) - "function words"
// 2. Context-specific phrases (used in few contexts) - "content words"
//
// Methods:
// - Generality Score: Contexts containing phrase / Total contexts
// - Shannon Entropy: Distribution evenness across contexts
// - Permutation Test: Statistical significance vs random chance

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WithinCallAnalysis {
    file_name: String,
    phrases: Vec<PhraseCandidate>,
    phrase_types: Vec<i32>,
    n_phrase_types: usize,
    stats: WithinCallStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseCandidate {
    id: usize,
    features: Vec<f64>,
    phrase_type: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WithinCallStats {
    n_phrases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileLabel {
    primary: String,
    labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GeneralityMetrics {
    phrase_id: i32,
    total_occurrences: usize,
    contexts_used: usize,
    total_contexts: usize,
    generality_score: f64,
    shannon_entropy: f64,
    normalized_entropy: f64,
    classification: PhraseType,
    context_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum PhraseType {
    UniversalGeneralist,
    Generalist,
    FlexibleSpecialist,
    ContextSpecialist,
    HighlySpecific,
    Rare,
}

#[derive(Debug, Clone, Serialize)]
struct PermutationTestResult {
    observed_mean_generality: f64,
    null_mean_generality: f64,
    null_std_generality: f64,
    p_value: f64,
    z_score: f64,
    significant: bool,
    n_permutations: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PhraseContextMatrix {
    matrix: HashMap<i32, HashMap<String, usize>>,
    phrase_totals: HashMap<i32, usize>,
    context_totals: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
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
}

#[derive(Debug, Clone, Serialize)]
struct AnalysisResults {
    metadata: Metadata,
    phrase_context_matrix: PhraseContextMatrix,
    generality_metrics: Vec<GeneralityMetrics>,
    permutation_test: PermutationTestResult,
    summary_statistics: SummaryStatistics,
}

#[derive(Debug, Clone, Serialize)]
struct Metadata {
    dataset: String,
    n_phrases: usize,
    n_contexts: usize,
    total_observations: usize,
    analysis_timestamp: String,
}

// ============================================================================
// Label Mappings
// ============================================================================

fn get_label_meanings() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("cc".to_string(), "Close Call".to_string());
    m.insert("sn".to_string(), "Sentinel".to_string());
    m.insert("soc".to_string(), "Social".to_string());
    m.insert("oth".to_string(), "Other".to_string());
    m.insert("agg".to_string(), "Aggression".to_string());
    m.insert("synch".to_string(), "Synchronized".to_string());
    m.insert("al".to_string(), "Alarm".to_string());
    m.insert("eating".to_string(), "Eating".to_string());
    m.insert("mo".to_string(), "Movement".to_string());
    m.insert("beep".to_string(), "Calibration".to_string());
    m.insert("ld".to_string(), "Lead".to_string());
    m
}

// ============================================================================
// Loading Functions
// ============================================================================

fn load_within_call_results(
    path: &Path,
) -> Result<Vec<WithinCallAnalysis>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let results: Vec<WithinCallAnalysis> = serde_json::from_reader(reader)?;
    Ok(results)
}

fn load_hdf5_labels(
    labels_dir: &Path,
) -> Result<HashMap<String, FileLabel>, Box<dyn std::error::Error>> {
    // Use external Python to read HDF5 since Rust HDF5 support is limited
    // We'll parse the output from a Python script

    let python_script = r#"
import h5py
import os
import json
import sys

labels_dir = sys.argv[1]
output = {}

for lbl_file in os.listdir(labels_dir):
    if not lbl_file.endswith('.h5'):
        continue

    fname = lbl_file.replace('.h5', '')
    path = os.path.join(labels_dir, lbl_file)

    try:
        with h5py.File(path, 'r') as f:
            lbls = f['lbl'][:]
            if len(lbls) > 0:
                lbls_str = [l.decode() if isinstance(l, bytes) else l for l in lbls]
                from collections import Counter
                lbl_counter = Counter(lbls_str)
                primary = lbl_counter.most_common(1)[0][0]

                output[fname] = {
                    'primary': primary,
                    'labels': lbls_str
                }
    except:
        pass

print(json.dumps(output))
"#;

    // Write script to temp file
    let script_path = "/tmp/load_meerkat_labels.py";
    std::fs::write(script_path, python_script)?;

    // Run Python script
    let output = std::process::Command::new("python3")
        .arg(script_path)
        .arg(labels_dir.to_str().unwrap())
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let labels: HashMap<String, FileLabel> = serde_json::from_str(&stdout)?;

    Ok(labels)
}

// ============================================================================
// Phrase-Context Matrix Building
// ============================================================================

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        (dot / (mag_a * mag_b)).max(0.0).min(1.0)
    } else {
        0.0
    }
}

fn cluster_phrases(
    analyses: &[WithinCallAnalysis],
    labels: &HashMap<String, FileLabel>,
    threshold: f64,
) -> (HashMap<i32, Vec<(String, String)>>, Vec<Vec<f64>>) {
    // Collect all phrase features with their file context
    let mut all_features: Vec<(Vec<f64>, String, String)> = Vec::new();

    for analysis in analyses {
        let fname = analysis.file_name.replace(".wav", "");
        if let Some(label) = labels.get(&fname) {
            for phrase in &analysis.phrases {
                if !phrase.features.is_empty() {
                    all_features.push((
                        phrase.features.clone(),
                        fname.clone(),
                        label.primary.clone(),
                    ));
                }
            }
        }
    }

    println!("   Total phrases with labels: {}", all_features.len());

    // Greedy clustering
    let mut type_representatives: Vec<Vec<f64>> = Vec::new();
    let mut phrase_to_type: HashMap<usize, i32> = HashMap::new();
    let mut type_assignments: HashMap<i32, Vec<(String, String)>> = HashMap::new();

    for (i, (features, fname, context)) in all_features.iter().enumerate() {
        let mut best_type = -1i32;
        let mut best_sim = 0.0;

        for (t, rep) in type_representatives.iter().enumerate() {
            let sim = cosine_similarity(features, rep);
            if sim >= threshold && sim > best_sim {
                best_sim = sim;
                best_type = t as i32;
            }
        }

        let assigned_type = if best_type >= 0 {
            best_type
        } else {
            let new_type = type_representatives.len() as i32;
            type_representatives.push(features.clone());
            new_type
        };

        phrase_to_type.insert(i, assigned_type);
        type_assignments
            .entry(assigned_type)
            .or_default()
            .push((fname.clone(), context.clone()));
    }

    println!("   Discovered {} phrase types", type_representatives.len());

    (type_assignments, type_representatives)
}

fn build_phrase_context_matrix(
    type_assignments: &HashMap<i32, Vec<(String, String)>>,
) -> PhraseContextMatrix {
    let mut matrix: HashMap<i32, HashMap<String, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<i32, usize> = HashMap::new();
    let mut context_totals: HashMap<String, usize> = HashMap::new();

    for (ptype, occurrences) in type_assignments {
        let phrase_total = occurrences.len();
        phrase_totals.insert(*ptype, phrase_total);

        let mut context_counts: HashMap<String, usize> = HashMap::new();
        for (_, context) in occurrences {
            let count = context_counts.entry(context.clone()).or_insert(0);
            *count += 1;
            let ctx_total = context_totals.entry(context.clone()).or_insert(0);
            *ctx_total += 1;
        }

        matrix.insert(*ptype, context_counts);
    }

    PhraseContextMatrix {
        matrix,
        phrase_totals,
        context_totals,
    }
}

// ============================================================================
// Generality Metrics
// ============================================================================

fn shannon_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn calculate_generality_metrics(
    pcm: &PhraseContextMatrix,
    n_contexts: usize,
) -> Vec<GeneralityMetrics> {
    let mut metrics = Vec::new();

    for (phrase_id, context_counts) in &pcm.matrix {
        let total_occurrences = pcm.phrase_totals.get(phrase_id).copied().unwrap_or(0);
        let contexts_used = context_counts.len();
        let counts: Vec<usize> = context_counts.values().copied().collect();

        let generality_score = contexts_used as f64 / n_contexts as f64;
        let entropy = shannon_entropy(&counts);
        let max_entropy = if contexts_used > 1 {
            (contexts_used as f64).log2()
        } else {
            0.0
        };
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        let classification = classify_phrase(generality_score, total_occurrences);

        metrics.push(GeneralityMetrics {
            phrase_id: *phrase_id,
            total_occurrences,
            contexts_used,
            total_contexts: n_contexts,
            generality_score,
            shannon_entropy: entropy,
            normalized_entropy,
            classification,
            context_distribution: context_counts.clone(),
        });
    }

    // Sort by generality score
    metrics.sort_by(|a, b| b.generality_score.partial_cmp(&a.generality_score).unwrap());

    metrics
}

fn classify_phrase(generality: f64, total: usize) -> PhraseType {
    if total < 5 {
        PhraseType::Rare
    } else if generality >= 0.8 {
        PhraseType::UniversalGeneralist
    } else if generality >= 0.5 {
        PhraseType::Generalist
    } else if generality >= 0.25 {
        PhraseType::FlexibleSpecialist
    } else if generality >= 0.1 {
        PhraseType::ContextSpecialist
    } else {
        PhraseType::HighlySpecific
    }
}

// ============================================================================
// Permutation Test
// ============================================================================

fn run_permutation_test(
    pcm: &PhraseContextMatrix,
    n_permutations: usize,
    n_contexts: usize,
) -> PermutationTestResult {
    // Calculate observed mean generality
    let observed_generalities: Vec<f64> = pcm
        .matrix
        .values()
        .map(|ctx_counts| ctx_counts.len() as f64 / n_contexts as f64)
        .collect();
    let observed_mean =
        observed_generalities.iter().sum::<f64>() / observed_generalities.len() as f64;

    // Collect all context assignments
    let all_contexts: Vec<String> = pcm
        .matrix
        .values()
        .flat_map(|ctx_counts| {
            ctx_counts
                .iter()
                .flat_map(|(ctx, &count)| std::iter::repeat(ctx.clone()).take(count))
                .collect::<Vec<_>>()
        })
        .collect();

    let context_names: Vec<String> = pcm.context_totals.keys().cloned().collect();
    let total_obs = all_contexts.len();

    // Run permutations
    let mut null_means: Vec<f64> = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Shuffle contexts
        let mut shuffled = all_contexts.clone();
        shuffle(&mut shuffled);

        // Calculate permuted generality
        let mut perm_generalities: Vec<f64> = Vec::new();
        let mut idx = 0;

        for (_, &phrase_total) in pcm.phrase_totals.iter() {
            let unique_contexts: HashSet<&String> = shuffled[idx..idx + phrase_total]
                .iter()
                .map(|s| s)
                .collect();
            perm_generalities.push(unique_contexts.len() as f64 / n_contexts as f64);
            idx += phrase_total;
        }

        null_means.push(perm_generalities.iter().sum::<f64>() / perm_generalities.len() as f64);
    }

    let null_mean = null_means.iter().sum::<f64>() / null_means.len() as f64;
    let null_var: f64 = null_means
        .iter()
        .map(|x| (x - null_mean).powi(2))
        .sum::<f64>()
        / null_means.len() as f64;
    let null_std = null_var.sqrt();

    let z_score = (observed_mean - null_mean) / null_std.max(1e-10);
    let p_value =
        null_means.iter().filter(|&&x| x >= observed_mean).count() as f64 / n_permutations as f64;

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

fn shuffle<T: Clone>(slice: &mut [T]) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    for i in (1..slice.len()).rev() {
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        let j = (hasher.finish() as usize) % (i + 1);
        slice.swap(i, j);
    }
}

// ============================================================================
// Summary Statistics
// ============================================================================

fn count_phrase_types(metrics: &[GeneralityMetrics]) -> (usize, usize, usize, usize, usize) {
    let mut universal = 0;
    let mut generalist = 0;
    let mut flexible = 0;
    let mut specialist = 0;
    let mut specific = 0;

    for m in metrics {
        match m.classification {
            PhraseType::UniversalGeneralist => universal += 1,
            PhraseType::Generalist => generalist += 1,
            PhraseType::FlexibleSpecialist => flexible += 1,
            PhraseType::ContextSpecialist => specialist += 1,
            PhraseType::HighlySpecific => specific += 1,
            PhraseType::Rare => {}
        }
    }

    (universal, generalist, flexible, specialist, specific)
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Phrase-Context Matrix Analysis - Meerkat Vocalizations      ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  HYPOTHESIS TESTING: Combinatorial Syntax vs Holistic Signals  ║");
    println!("║                                                                 ║");
    println!("║  If combinatorial syntax exists:                                ║");
    println!("║    • General-purpose phrases - \"function words\"                  ║");
    println!("║    • Context-specific phrases - \"content words\"                   ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12");
    let results_path = data_dir.join("within_call_results/meerkat_within_call_analyses.json");
    let labels_dir = data_dir.join("lbl/08000Hz");
    let output_dir = data_dir.join("phrase_context_results");

    fs::create_dir_all(&output_dir)?;

    // ========================================================================
    // Step 1: Load Data
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Data                                             │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📂 Loading within-call phrase analysis results...");
    let analyses = load_within_call_results(&results_path)?;
    println!("      └─ Loaded {} vocalization analyses", analyses.len());
    println!();

    println!("   📂 Loading behavioral context labels (HDF5)...");
    let labels = load_hdf5_labels(&labels_dir)?;
    println!("      └─ Loaded {} files with labels", labels.len());
    println!();

    // ========================================================================
    // Step 2: Cluster Phrases and Build Matrix
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Clustering Phrases and Building Matrix                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   🔄 Clustering phrases using cosine similarity (threshold 0.75)...");
    let (type_assignments, _) = cluster_phrases(&analyses, &labels, 0.75);
    println!();

    println!("   📊 Building phrase-context matrix...");
    let pcm = build_phrase_context_matrix(&type_assignments);

    let n_phrases = pcm.matrix.len();
    let n_contexts = pcm.context_totals.len();
    let total_obs: usize = pcm.phrase_totals.values().sum();

    println!("      ├─ Unique phrases: {}", n_phrases);
    println!("      ├─ Behavioral contexts: {}", n_contexts);
    println!("      └─ Total observations: {}", total_obs);
    println!();

    // Display context distribution
    println!("   📋 Context Distribution:");
    let mut context_vec: Vec<_> = pcm.context_totals.iter().collect();
    context_vec.sort_by(|a, b| b.1.cmp(a.1));
    let label_meanings = get_label_meanings();
    for (ctx, count) in context_vec.iter() {
        let meaning = label_meanings.get(*ctx).unwrap_or(ctx);
        println!(
            "      {} ({}): {} observations ({:.1}%)",
            ctx,
            meaning,
            count,
            **count as f64 / total_obs as f64 * 100.0
        );
    }
    println!();

    // ========================================================================
    // Step 3: Calculate Generality Metrics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Calculating Generality Metrics                          │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let metrics = calculate_generality_metrics(&pcm, n_contexts);
    println!("      └─ Computed metrics for {} phrases", metrics.len());
    println!();

    // ========================================================================
    // Step 4: Phrase Type Classification
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Phrase Type Classification                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let type_counts = count_phrase_types(&metrics);
    println!("   🏷️  Phrase Type Distribution:");
    println!("      ┌────────────────────────────┬──────────┬──────────┐");
    println!("      │ Type                       │ Count    │ Percentage│");
    println!("      ├────────────────────────────┼──────────┼──────────┤");
    println!(
        "      │ Universal Generalist       │ {:8} │ {:8.1}%│",
        type_counts.0,
        type_counts.0 as f64 / metrics.len() as f64 * 100.0
    );
    println!(
        "      │ Generalist                 │ {:8} │ {:8.1}%│",
        type_counts.1,
        type_counts.1 as f64 / metrics.len() as f64 * 100.0
    );
    println!(
        "      │ Flexible Specialist        │ {:8} │ {:8.1}%│",
        type_counts.2,
        type_counts.2 as f64 / metrics.len() as f64 * 100.0
    );
    println!(
        "      │ Context Specialist         │ {:8} │ {:8.1}%│",
        type_counts.3,
        type_counts.3 as f64 / metrics.len() as f64 * 100.0
    );
    println!(
        "      │ Highly Specific            │ {:8} │ {:8.1}%│",
        type_counts.4,
        type_counts.4 as f64 / metrics.len() as f64 * 100.0
    );
    println!("      └────────────────────────────┴──────────┴──────────┘");
    println!();

    // Display top phrases by generality
    println!("   📈 Top 10 General-Purpose Phrases:");
    println!("      ┌─────────┬────────────┬───────────┬────────────────────────────────┐");
    println!("      │ Type ID │ Generality │ Occurs    │ Top Contexts                   │");
    println!("      ├─────────┼────────────┼───────────┼────────────────────────────────┤");
    for m in metrics.iter().take(10) {
        let mut ctx_vec: Vec<_> = m.context_distribution.iter().collect();
        ctx_vec.sort_by(|a, b| b.1.cmp(a.1));
        let ctx_str: String = ctx_vec
            .iter()
            .take(3)
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "      │ {:7} │ {:10.3} │ {:9} │ {:30} │",
            m.phrase_id, m.generality_score, m.total_occurrences, ctx_str
        );
    }
    println!("      └─────────┴────────────┴───────────┴────────────────────────────────┘");
    println!();

    // ========================================================================
    // Step 5: Permutation Test
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Permutation Test (Statistical Significance)            │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let n_permutations = 500;
    println!("      ├─ Number of permutations: {}", n_permutations);
    println!("      └─ Running test...");

    let perm_result = run_permutation_test(&pcm, n_permutations, n_contexts);

    println!();
    println!("   ✅ Permutation Test Results:");
    println!(
        "      ├─ Observed mean generality: {:.4}",
        perm_result.observed_mean_generality
    );
    println!(
        "      ├─ Null mean generality:      {:.4} ± {:.4}",
        perm_result.null_mean_generality, perm_result.null_std_generality
    );
    println!(
        "      ├─ Z-score:                   {:.4}",
        perm_result.z_score
    );
    println!(
        "      ├─ P-value:                   {:.4}",
        perm_result.p_value
    );
    println!(
        "      └─ Significant:               {}",
        if perm_result.significant {
            "YES ✓"
        } else {
            "NO ✗"
        }
    );
    println!();

    // ========================================================================
    // Step 6: Summary Statistics
    // ========================================================================

    let generality_scores: Vec<f64> = metrics.iter().map(|m| m.generality_score).collect();
    let mean_generality = generality_scores.iter().sum::<f64>() / generality_scores.len() as f64;
    let median_generality = {
        let mut sorted = generality_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let std_generality = {
        let mean = mean_generality;
        let var: f64 = generality_scores
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / generality_scores.len() as f64;
        var.sqrt()
    };

    let entropy_scores: Vec<f64> = metrics.iter().map(|m| m.shannon_entropy).collect();
    let mean_entropy = entropy_scores.iter().sum::<f64>() / entropy_scores.len() as f64;

    let summary = SummaryStatistics {
        n_universal_phrases: type_counts.0,
        n_generalist_phrases: type_counts.1,
        n_flexible_specialist_phrases: type_counts.2,
        n_context_specialist_phrases: type_counts.3,
        n_highly_specific_phrases: type_counts.4,
        mean_generality_score: mean_generality,
        median_generality_score: median_generality,
        std_generality_score: std_generality,
        mean_shannon_entropy: mean_entropy,
    };

    // ========================================================================
    // Step 7: Hypothesis Conclusion
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ HYPOTHESIS TEST CONCLUSION                                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let pct_general = (type_counts.0 + type_counts.1) as f64 / metrics.len() as f64 * 100.0;
    let pct_specific = (type_counts.3 + type_counts.4) as f64 / metrics.len() as f64 * 100.0;

    println!("   📊 Key Metrics:");
    println!("      ├─ Mean Generality Score:    {:.3}", mean_generality);
    println!(
        "      ├─ Median Generality Score:  {:.3}",
        median_generality
    );
    println!("      ├─ General Phrases:          {:.1}%", pct_general);
    println!("      ├─ Specific Phrases:         {:.1}%", pct_specific);
    println!(
        "      └─ Permutation p-value:      {:.4}",
        perm_result.p_value
    );
    println!();

    if pct_general > 20.0 && perm_result.significant {
        println!("   ✅ EVIDENCE SUPPORTS COMBINATORIAL SYNTAX HYPOTHESIS");
        println!();
        println!("      A significant portion of phrases are used across multiple contexts,");
        println!("      suggesting they function as general-purpose building blocks.");
        println!("      This is consistent with a \"sentence structure\" where:");
        println!("        • General phrases = Function words (grammar/syntax)");
        println!("        • Specific phrases = Content words (meaning/semantics)");
    } else if pct_general > 10.0 {
        println!("   ~ PARTIAL EVIDENCE FOR COMBINATORIAL SYNTAX");
        println!();
        println!("      Some phrases show cross-context usage, but the majority are");
        println!("      context-specific. This could indicate a limited combinatorial system.");
    } else {
        println!("   ❌ EVIDENCE FAVORS HOLISTIC/REFLEXIVE HYPOTHESIS");
        println!();
        println!("      Most phrases are highly context-specific, suggesting fixed signals");
        println!("      for specific behaviors rather than reusable building blocks.");
    }
    println!();

    // ========================================================================
    // Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let results = AnalysisResults {
        metadata: Metadata {
            dataset: "Meerkat Vocalizations (8kHz)".to_string(),
            n_phrases,
            n_contexts,
            total_observations: total_obs,
            analysis_timestamp: chrono::Utc::now().to_rfc3339(),
        },
        phrase_context_matrix: pcm.clone(),
        generality_metrics: metrics.into_iter().take(1000).collect(),
        permutation_test: perm_result,
        summary_statistics: summary,
    };

    let output_path = output_dir.join("meerkat_phrase_context_analysis.json");
    let file = File::create(&output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &results)?;
    println!("   ✓ Saved to: {:?}", output_path);

    // Save CSV of phrase-context matrix
    let csv_path = output_dir.join("phrase_context_matrix.csv");
    let mut csv_content = String::new();
    csv_content.push_str("phrase_type,total,generality,classification,context_distribution\n");
    for (ptype, ctx_counts) in &pcm.matrix {
        let total = pcm.phrase_totals.get(ptype).copied().unwrap_or(0);
        let gen = ctx_counts.len() as f64 / n_contexts as f64;
        let ctx_str: String = ctx_counts
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(";");
        csv_content.push_str(&format!("{},{},{:.3},\"{}\"\n", ptype, total, gen, ctx_str));
    }
    std::fs::write(&csv_path, csv_content)?;
    println!("   ✓ Matrix saved to: {:?}", csv_path);

    let elapsed = start_time.elapsed();
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!(
        "║ ANALYSIS COMPLETE ({:.1?})                                      ║",
        elapsed
    );
    println!("╚═════════════════════════════════════════════════════════════════╝");

    Ok(())
}
