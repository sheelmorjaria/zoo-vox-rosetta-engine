// Combined Analysis: Within-Call Patterns × Behavioral Contexts
//
// This analysis combines:
// 1. Within-call phrase discovery results (local phrase sequences)
// 2. Behavioral context annotations
//
// Questions we answer:
// - Do certain contexts favor more complex vocabulary (higher entropy)?
// - Are repetition patterns (motifs) context-specific?
// - Which contexts have the longest/shortest calls?
// - Is syllable repetition associated with specific behaviors?
//
// Usage: cargo run --release --example bat_within_call_context_analysis

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    #[serde(rename = "Emitter")]
    emitter: i32,
    #[serde(rename = "Addressee")]
    addressee: i32,
    #[serde(rename = "Context")]
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

#[derive(Debug, Clone, Deserialize)]
struct WithinCallAnalysis {
    file_name: String,
    call_type: Option<String>,
    total_duration_ms: f64,
    phrases: Vec<PhraseCandidate>,
    n_phrase_types: usize,
    phrase_types: Vec<i32>,
    motifs: Vec<Motif>,
    stats: WithinCallStats,
}

#[derive(Debug, Clone, Deserialize)]
struct PhraseCandidate {
    id: usize,
    start_ms: f64,
    end_ms: f64,
    duration_ms: f64,
    phrase_type: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
struct Motif {
    pattern: Vec<i32>,
    occurrences: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct WithinCallStats {
    n_phrases: usize,
    avg_phrase_duration_ms: f64,
    type_entropy: f64,
    phrase_rate: f64,
}

// =============================================================================
// Analysis Results
// =============================================================================

#[derive(Debug, Clone, Serialize)]
struct CombinedAnalysisResults {
    metadata: AnalysisMetadata,
    context_statistics: Vec<ContextStatistics>,
    repetition_analysis: RepetitionAnalysis,
    vocabulary_complexity: VocabularyComplexityAnalysis,
    duration_analysis: DurationAnalysis,
    phrase_rate_analysis: PhraseRateAnalysis,
    motif_context_association: MotifContextAssociation,
    key_findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AnalysisMetadata {
    total_vocalizations: usize,
    total_with_annotations: usize,
    unique_contexts: usize,
    total_phrases: usize,
    total_motifs: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ContextStatistics {
    context_id: i32,
    n_vocalizations: usize,
    n_phrases: usize,
    n_motifs: usize,
    avg_phrases_per_call: f64,
    avg_duration_ms: f64,
    avg_entropy: f64,
    avg_phrase_rate: f64,
    pct_with_motifs: f64,
    avg_motif_length: f64,
    repetition_patterns: HashMap<String, usize>,  // pattern -> count
}

#[derive(Debug, Clone, Serialize)]
struct RepetitionAnalysis {
    // Does context influence syllable repetition?
    context_repetition_scores: Vec<ContextRepetitionScore>,
    overall_repetition_rate: f64,
    most_repetitive_context: Option<i32>,
    least_repetitive_context: Option<i32>,
    statistical_summary: RepetitionStats,
}

#[derive(Debug, Clone, Serialize)]
struct ContextRepetitionScore {
    context_id: i32,
    n_calls: usize,
    n_with_repetition: usize,
    repetition_rate: f64,
    avg_repetition_count: f64,  // avg motifs per call
}

#[derive(Debug, Clone, Serialize)]
struct RepetitionStats {
    chi_square: f64,
    p_value: f64,
    significant: bool,
    interpretation: String,
}

#[derive(Debug, Clone, Serialize)]
struct VocabularyComplexityAnalysis {
    // Which contexts have richer vocabulary?
    context_entropy_ranking: Vec<ContextEntropyRanking>,
    simple_calls_by_context: HashMap<i32, usize>,  // entropy < 0.3
    complex_calls_by_context: HashMap<i32, usize>, // entropy >= 0.3
    interpretation: String,
}

#[derive(Debug, Clone, Serialize)]
struct ContextEntropyRanking {
    context_id: i32,
    n_calls: usize,
    mean_entropy: f64,
    median_entropy: f64,
    max_entropy: f64,
    pct_simple: f64,
    pct_complex: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DurationAnalysis {
    // Call duration by context
    context_durations: Vec<ContextDuration>,
    longest_context: Option<i32>,
    shortest_context: Option<i32>,
    duration_variance_explained: f64,  // How much does context explain duration?
}

#[derive(Debug, Clone, Serialize)]
struct ContextDuration {
    context_id: i32,
    n_calls: usize,
    mean_ms: f64,
    median_ms: f64,
    std_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PhraseRateAnalysis {
    // Phrase rate (syllables/sec) by context
    context_rates: Vec<ContextPhraseRate>,
    fastest_context: Option<i32>,
    slowest_context: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct ContextPhraseRate {
    context_id: i32,
    mean_rate: f64,
    median_rate: f64,
    std_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MotifContextAssociation {
    // Which motif patterns are associated with which contexts?
    pattern_context_distribution: HashMap<String, Vec<(i32, f64)>>,  // pattern -> [(context, percentage)]
    context_specific_patterns: HashMap<i32, Vec<String>>,  // context -> patterns that are overrepresented
    universal_patterns: Vec<String>,  // patterns found across all contexts
}

// =============================================================================
// Analysis Functions
// =============================================================================

fn load_annotations(path: &PathBuf) -> Result<Vec<Annotation>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));

    let mut annotations = Vec::new();
    for result in reader.deserialize() {
        let annotation: Annotation = result?;
        annotations.push(annotation);
    }

    Ok(annotations)
}

fn load_within_call_results(path: &PathBuf) -> Result<Vec<WithinCallAnalysis>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let results = serde_json::from_reader(reader)?;
    Ok(results)
}

fn analyze_combined(
    analyses: &[WithinCallAnalysis],
    annotations: &[Annotation],
) -> CombinedAnalysisResults {
    // Build file_name -> context mapping
    let file_context_map: HashMap<String, i32> = annotations
        .iter()
        .map(|a| (a.file_name.trim().to_string(), a.context))
        .collect();

    // Group analyses by context
    let mut by_context: HashMap<i32, Vec<&WithinCallAnalysis>> = HashMap::new();
    let mut total_with_annotations = 0;
    let mut total_phrases = 0;
    let mut total_motifs = 0;

    for analysis in analyses {
        // Try to match file name - annotations include .wav extension
        let file_name = analysis.file_name.trim();

        if let Some(&context) = file_context_map.get(file_name) {
            by_context.entry(context).or_default().push(analysis);
            total_with_annotations += 1;
            total_phrases += analysis.stats.n_phrases;
            total_motifs += analysis.motifs.len();
        }
    }

    let unique_contexts = by_context.len();

    // Context statistics
    let context_statistics = compute_context_statistics(&by_context);

    // Repetition analysis
    let repetition_analysis = analyze_repetition(&by_context);

    // Vocabulary complexity
    let vocabulary_complexity = analyze_vocabulary_complexity(&by_context);

    // Duration analysis
    let duration_analysis = analyze_duration(&by_context);

    // Phrase rate analysis
    let phrase_rate_analysis = analyze_phrase_rate(&by_context);

    // Motif-context association
    let motif_context_association = analyze_motif_context_association(&by_context);

    // Key findings
    let key_findings = generate_key_findings(
        &context_statistics,
        &repetition_analysis,
        &vocabulary_complexity,
        &duration_analysis,
    );

    CombinedAnalysisResults {
        metadata: AnalysisMetadata {
            total_vocalizations: analyses.len(),
            total_with_annotations,
            unique_contexts,
            total_phrases,
            total_motifs,
        },
        context_statistics,
        repetition_analysis,
        vocabulary_complexity,
        duration_analysis,
        phrase_rate_analysis,
        motif_context_association,
        key_findings,
    }
}

fn compute_context_statistics(by_context: &HashMap<i32, Vec<&WithinCallAnalysis>>) -> Vec<ContextStatistics> {
    let mut stats = Vec::new();

    for (&context_id, analyses) in by_context {
        let n_vocalizations = analyses.len();
        let n_phrases: usize = analyses.iter().map(|a| a.stats.n_phrases).sum();
        let n_motifs: usize = analyses.iter().map(|a| a.motifs.len()).sum();

        let avg_phrases_per_call = n_phrases as f64 / n_vocalizations as f64;
        let avg_duration_ms = analyses.iter()
            .map(|a| a.total_duration_ms)
            .sum::<f64>() / n_vocalizations as f64;

        let analyses_with_phrases: Vec<_> = analyses.iter()
            .filter(|a| a.stats.n_phrases > 0)
            .collect();

        let avg_entropy = if !analyses_with_phrases.is_empty() {
            analyses_with_phrases.iter()
                .map(|a| a.stats.type_entropy)
                .sum::<f64>() / analyses_with_phrases.len() as f64
        } else {
            0.0
        };

        let avg_phrase_rate = if !analyses_with_phrases.is_empty() {
            analyses_with_phrases.iter()
                .map(|a| a.stats.phrase_rate)
                .sum::<f64>() / analyses_with_phrases.len() as f64
        } else {
            0.0
        };

        let with_motifs = analyses.iter().filter(|a| !a.motifs.is_empty()).count();
        let pct_with_motifs = with_motifs as f64 / n_vocalizations as f64 * 100.0;

        // Average motif length
        let total_motif_len: usize = analyses.iter()
            .flat_map(|a| a.motifs.iter())
            .map(|m| m.pattern.len())
            .sum();
        let total_motif_count: usize = analyses.iter()
            .flat_map(|a| a.motifs.iter())
            .map(|m| m.occurrences)
            .sum();
        let avg_motif_length = if total_motif_count > 0 {
            total_motif_len as f64 / analyses.iter()
                .flat_map(|a| a.motifs.iter())
                .count() as f64
        } else {
            0.0
        };

        // Repetition patterns (simplified: count [0-0], [0-0-0], etc.)
        let mut repetition_patterns: HashMap<String, usize> = HashMap::new();
        for analysis in analyses {
            for motif in &analysis.motifs {
                let pattern_str = motif.pattern.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join("-");
                *repetition_patterns.entry(pattern_str).or_default() += motif.occurrences;
            }
        }

        stats.push(ContextStatistics {
            context_id,
            n_vocalizations,
            n_phrases,
            n_motifs,
            avg_phrases_per_call,
            avg_duration_ms,
            avg_entropy,
            avg_phrase_rate,
            pct_with_motifs,
            avg_motif_length,
            repetition_patterns,
        });
    }

    stats.sort_by(|a, b| b.n_vocalizations.cmp(&a.n_vocalizations));
    stats
}

fn analyze_repetition(by_context: &HashMap<i32, Vec<&WithinCallAnalysis>>) -> RepetitionAnalysis {
    let mut context_scores: Vec<ContextRepetitionScore> = Vec::new();

    for (&context_id, analyses) in by_context {
        let n_calls = analyses.len();
        let n_with_repetition = analyses.iter()
            .filter(|a| !a.motifs.is_empty())
            .count();

        let total_motifs: usize = analyses.iter()
            .map(|a| a.motifs.len())
            .sum();

        context_scores.push(ContextRepetitionScore {
            context_id,
            n_calls,
            n_with_repetition,
            repetition_rate: n_with_repetition as f64 / n_calls as f64 * 100.0,
            avg_repetition_count: total_motifs as f64 / n_calls as f64,
        });
    }

    context_scores.sort_by(|a, b| b.repetition_rate.partial_cmp(&a.repetition_rate).unwrap());

    let overall_repetition_rate = context_scores.iter()
        .map(|s| s.n_with_repetition as f64)
        .sum::<f64>() / context_scores.iter().map(|s| s.n_calls as f64).sum::<f64>() * 100.0;

    let most_repetitive = context_scores.first().map(|s| s.context_id);
    let least_repetitive = context_scores.last().map(|s| s.context_id);

    // Chi-square test for context vs repetition
    let total_calls: usize = context_scores.iter().map(|s| s.n_calls).sum();
    let total_with_rep: usize = context_scores.iter().map(|s| s.n_with_repetition).sum();

    let expected_rate = total_with_rep as f64 / total_calls as f64;
    let mut chi_square = 0.0;

    for score in &context_scores {
        let expected = score.n_calls as f64 * expected_rate;
        let observed = score.n_with_repetition as f64;
        if expected > 0.0 {
            chi_square += (observed - expected).powi(2) / expected;
        }
    }

    // Simplified p-value estimation (chi-square with df = n_contexts - 1)
    let df = (context_scores.len() - 1) as f64;
    let p_value = estimate_p_value_chi_square(chi_square, df);
    let significant = p_value < 0.05;

    let interpretation = if significant {
        format!(
            "Context significantly influences repetition (χ²={:.2}, p={:.4}). \
             Context {} has highest repetition ({:.1}%), context {} lowest ({:.1}%).",
            chi_square, p_value,
            most_repetitive.unwrap_or(-1),
            context_scores.first().map(|s| s.repetition_rate).unwrap_or(0.0),
            least_repetitive.unwrap_or(-1),
            context_scores.last().map(|s| s.repetition_rate).unwrap_or(0.0)
        )
    } else {
        format!(
            "No significant context effect on repetition (χ²={:.2}, p={:.4}). \
             Repetition is consistent across behavioral contexts.",
            chi_square, p_value
        )
    };

    RepetitionAnalysis {
        context_repetition_scores: context_scores,
        overall_repetition_rate,
        most_repetitive_context: most_repetitive,
        least_repetitive_context: least_repetitive,
        statistical_summary: RepetitionStats {
            chi_square,
            p_value,
            significant,
            interpretation,
        },
    }
}

fn estimate_p_value_chi_square(chi_sq: f64, df: f64) -> f64 {
    // Simplified chi-square p-value estimation
    // Using Wilson-Hilferty approximation for larger df
    if df <= 0.0 || chi_sq <= 0.0 {
        return 1.0;
    }

    let z = ((chi_sq / df).powf(1.0/3.0) - (1.0 - 2.0/(9.0*df))) / (2.0/(9.0*df)).sqrt();

    // Standard normal CDF approximation
    let p = 0.5 * (1.0 + erf(z / 2.0_f64.sqrt()));
    1.0 - p
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn analyze_vocabulary_complexity(by_context: &HashMap<i32, Vec<&WithinCallAnalysis>>) -> VocabularyComplexityAnalysis {
    let mut rankings: Vec<ContextEntropyRanking> = Vec::new();
    let mut simple_by_context: HashMap<i32, usize> = HashMap::new();
    let mut complex_by_context: HashMap<i32, usize> = HashMap::new();

    for (&context_id, analyses) in by_context {
        let entropies: Vec<f64> = analyses.iter()
            .filter(|a| a.stats.n_phrases > 1)
            .map(|a| a.stats.type_entropy)
            .collect();

        if entropies.is_empty() {
            continue;
        }

        let mean_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let mut sorted = entropies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_entropy = sorted[sorted.len() / 2];
        let max_entropy = sorted.last().copied().unwrap_or(0.0);

        let simple = entropies.iter().filter(|&&e| e < 0.3).count();
        let complex = entropies.iter().filter(|&&e| e >= 0.3).count();
        let total = entropies.len();

        rankings.push(ContextEntropyRanking {
            context_id,
            n_calls: entropies.len(),
            mean_entropy,
            median_entropy,
            max_entropy,
            pct_simple: simple as f64 / total as f64 * 100.0,
            pct_complex: complex as f64 / total as f64 * 100.0,
        });

        simple_by_context.insert(context_id, simple);
        complex_by_context.insert(context_id, complex);
    }

    rankings.sort_by(|a, b| b.mean_entropy.partial_cmp(&a.mean_entropy).unwrap());

    let interpretation = if rankings.iter().all(|r| r.mean_entropy < 0.1) {
        "All contexts show extremely low entropy - bats typically use one phrase type per call \
         regardless of behavioral context. This suggests syllable repetition rather than \
         syllable combination is the primary syntactic mechanism."
            .to_string()
    } else {
        format!(
            "Vocabulary complexity varies across contexts. Context {} shows highest complexity \
             (mean entropy {:.3}), context {} lowest ({:.3}).",
            rankings.first().map(|r| r.context_id).unwrap_or(-1),
            rankings.first().map(|r| r.mean_entropy).unwrap_or(0.0),
            rankings.last().map(|r| r.context_id).unwrap_or(-1),
            rankings.last().map(|r| r.mean_entropy).unwrap_or(0.0)
        )
    };

    VocabularyComplexityAnalysis {
        context_entropy_ranking: rankings,
        simple_calls_by_context: simple_by_context,
        complex_calls_by_context: complex_by_context,
        interpretation,
    }
}

fn analyze_duration(by_context: &HashMap<i32, Vec<&WithinCallAnalysis>>) -> DurationAnalysis {
    let mut durations: Vec<ContextDuration> = Vec::new();

    for (&context_id, analyses) in by_context {
        let duration_vals: Vec<f64> = analyses.iter()
            .map(|a| a.total_duration_ms)
            .collect();

        if duration_vals.is_empty() {
            continue;
        }

        let mean = duration_vals.iter().sum::<f64>() / duration_vals.len() as f64;
        let mut sorted = duration_vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = sorted[sorted.len() / 2];
        let variance = duration_vals.iter()
            .map(|d| (d - mean).powi(2))
            .sum::<f64>() / duration_vals.len() as f64;

        durations.push(ContextDuration {
            context_id,
            n_calls: duration_vals.len(),
            mean_ms: mean,
            median_ms: median,
            std_ms: variance.sqrt(),
            min_ms: sorted.first().copied().unwrap_or(0.0),
            max_ms: sorted.last().copied().unwrap_or(0.0),
        });
    }

    durations.sort_by(|a, b| b.mean_ms.partial_cmp(&a.mean_ms).unwrap());

    let longest = durations.first().map(|d| d.context_id);
    let shortest = durations.last().map(|d| d.context_id);

    // Simplified variance explained (ratio of between-group to total variance)
    let overall_mean: f64 = durations.iter()
        .map(|d| d.mean_ms * d.n_calls as f64)
        .sum::<f64>() / durations.iter().map(|d| d.n_calls as f64).sum::<f64>();

    let between_var: f64 = durations.iter()
        .map(|d| (d.mean_ms - overall_mean).powi(2) * d.n_calls as f64)
        .sum::<f64>() / durations.iter().map(|d| d.n_calls as f64).sum::<f64>();

    let avg_std: f64 = durations.iter().map(|d| d.std_ms).sum::<f64>() / durations.len() as f64;
    let total_var = avg_std.powi(2);
    let variance_explained = if total_var > 0.0 {
        between_var / (between_var + total_var) * 100.0
    } else {
        0.0
    };

    DurationAnalysis {
        context_durations: durations,
        longest_context: longest,
        shortest_context: shortest,
        duration_variance_explained: variance_explained,
    }
}

fn analyze_phrase_rate(by_context: &HashMap<i32, Vec<&WithinCallAnalysis>>) -> PhraseRateAnalysis {
    let mut rates: Vec<ContextPhraseRate> = Vec::new();

    for (&context_id, analyses) in by_context {
        let rate_vals: Vec<f64> = analyses.iter()
            .filter(|a| a.stats.n_phrases > 0)
            .map(|a| a.stats.phrase_rate)
            .collect();

        if rate_vals.is_empty() {
            continue;
        }

        let mean = rate_vals.iter().sum::<f64>() / rate_vals.len() as f64;
        let variance = rate_vals.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / rate_vals.len() as f64;

        let mut sorted = rate_vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];

        rates.push(ContextPhraseRate {
            context_id,
            mean_rate: mean,
            median_rate: median,
            std_rate: variance.sqrt(),
        });
    }

    rates.sort_by(|a, b| b.mean_rate.partial_cmp(&a.mean_rate).unwrap());

    let fastest = rates.first().map(|r| r.context_id);
    let slowest = rates.last().map(|r| r.context_id);

    PhraseRateAnalysis {
        context_rates: rates,
        fastest_context: fastest,
        slowest_context: slowest,
    }
}

fn analyze_motif_context_association(by_context: &HashMap<i32, Vec<&WithinCallAnalysis>>) -> MotifContextAssociation {
    let mut pattern_context_counts: HashMap<String, HashMap<i32, usize>> = HashMap::new();
    let mut total_by_context: HashMap<i32, usize> = HashMap::new();

    // Count patterns by context
    for (&context_id, analyses) in by_context {
        let mut context_total = 0;

        for analysis in analyses {
            for motif in &analysis.motifs {
                let pattern_str = motif.pattern.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join("-");

                *pattern_context_counts
                    .entry(pattern_str.clone())
                    .or_default()
                    .entry(context_id)
                    .or_default() += motif.occurrences;

                context_total += motif.occurrences;
            }
        }

        total_by_context.insert(context_id, context_total);
    }

    // Calculate percentage distribution for each pattern
    let mut pattern_context_distribution: HashMap<String, Vec<(i32, f64)>> = HashMap::new();

    for (pattern, context_counts) in &pattern_context_counts {
        let total: usize = context_counts.values().sum();
        if total < 10 {
            continue;  // Skip rare patterns
        }

        let distribution: Vec<(i32, f64)> = context_counts.iter()
            .map(|(&ctx, &count)| (ctx, count as f64 / total as f64 * 100.0))
            .collect();

        pattern_context_distribution.insert(pattern.clone(), distribution);
    }

    // Find context-specific patterns (>50% in one context)
    let mut context_specific: HashMap<i32, Vec<String>> = HashMap::new();
    for (pattern, dist) in &pattern_context_distribution {
        if let Some((dominant_ctx, dominant_pct)) = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            if *dominant_pct > 50.0 {
                context_specific.entry(*dominant_ctx).or_default().push(pattern.clone());
            }
        }
    }

    // Find universal patterns (present in all contexts with <30% max)
    let all_contexts: HashSet<i32> = by_context.keys().cloned().collect();
    let universal: Vec<String> = pattern_context_distribution.iter()
        .filter(|(_, dist)| {
            let contexts_present: HashSet<i32> = dist.iter().map(|(c, _)| *c).collect();
            let max_pct = dist.iter().map(|(_, p)| *p).fold(0.0, f64::max);
            contexts_present.len() >= all_contexts.len() / 2 && max_pct < 30.0
        })
        .map(|(p, _)| p.clone())
        .collect();

    MotifContextAssociation {
        pattern_context_distribution,
        context_specific_patterns: context_specific,
        universal_patterns: universal,
    }
}

fn generate_key_findings(
    context_stats: &[ContextStatistics],
    repetition: &RepetitionAnalysis,
    vocabulary: &VocabularyComplexityAnalysis,
    duration: &DurationAnalysis,
) -> Vec<String> {
    let mut findings = Vec::new();

    // Finding 1: Repetition
    if repetition.statistical_summary.significant {
        findings.push(format!(
            "📊 SYNTACTIC INSIGHT: Repetition patterns vary significantly by context (p={:.4}). \
             Context {} shows {:.1}% repetition rate vs {} at {:.1}%.",
            repetition.statistical_summary.p_value,
            repetition.most_repetitive_context.unwrap_or(-1),
            repetition.context_repetition_scores.first().map(|s| s.repetition_rate).unwrap_or(0.0),
            repetition.least_repetitive_context.unwrap_or(-1),
            repetition.context_repetition_scores.last().map(|s| s.repetition_rate).unwrap_or(0.0)
        ));
    } else {
        findings.push(format!(
            "📊 SYNTACTIC INSIGHT: Repetition is UNIVERSAL across contexts ({:.1}% overall). \
             Bats use syllable repetition consistently regardless of behavioral situation.",
            repetition.overall_repetition_rate
        ));
    }

    // Finding 2: Duration
    if let (Some(longest), Some(shortest)) = (duration.longest_context, duration.shortest_context) {
        let longest_dur = duration.context_durations.iter()
            .find(|d| d.context_id == longest)
            .map(|d| d.mean_ms)
            .unwrap_or(0.0);
        let shortest_dur = duration.context_durations.iter()
            .find(|d| d.context_id == shortest)
            .map(|d| d.mean_ms)
            .unwrap_or(0.0);

        findings.push(format!(
            "📊 DURATION: Context {} produces longest calls ({:.0}ms avg), \
             context {} shortest ({:.0}ms avg). {:.1}% of duration variance explained by context.",
            longest, longest_dur, shortest, shortest_dur, duration.duration_variance_explained
        ));
    }

    // Finding 3: Vocabulary
    findings.push(format!(
        "📊 VOCABULARY: {} interpretation: {}",
        vocabulary.interpretation.len().min(200),
        &vocabulary.interpretation[..vocabulary.interpretation.len().min(200)]
    ));

    // Finding 4: Most common context
    if let Some(top_context) = context_stats.first() {
        findings.push(format!(
            "📊 PREVALENCE: Context {} is most common ({:.1}% of vocalizations, {} calls). \
             Avg {:.1} phrases/call, {:.1}% with motifs.",
            top_context.context_id,
            top_context.n_vocalizations as f64 / context_stats.iter().map(|c| c.n_vocalizations).sum::<usize>() as f64 * 100.0,
            top_context.n_vocalizations,
            top_context.avg_phrases_per_call,
            top_context.pct_with_motifs
        ));
    }

    findings
}

// =============================================================================
// Output
// =============================================================================

impl CombinedAnalysisResults {
    fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║    Within-Call × Context Combined Analysis Summary             ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📊 METADATA");
        println!("   ├─ Total vocalizations: {:>12}", self.metadata.total_vocalizations);
        println!("   ├─ Matched with annotations: {:>8}", self.metadata.total_with_annotations);
        println!("   ├─ Unique contexts: {:>15}", self.metadata.unique_contexts);
        println!("   ├─ Total phrases: {:>16}", self.metadata.total_phrases);
        println!("   └─ Total motifs: {:>18}", self.metadata.total_motifs);

        println!("\n📊 CONTEXT STATISTICS (Top 10)");
        println!("   {:<10} {:>10} {:>12} {:>10} {:>12}",
                 "Context", "Calls", "Avg Phrases", "% Motifs", "Avg Duration");
        println!("   {}", "-".repeat(60));

        for ctx in self.context_statistics.iter().take(10) {
            println!("   {:<10} {:>10} {:>12.1} {:>9.1}% {:>10.0}ms",
                     ctx.context_id, ctx.n_vocalizations, ctx.avg_phrases_per_call,
                     ctx.pct_with_motifs, ctx.avg_duration_ms);
        }

        println!("\n📊 REPETITION ANALYSIS");
        println!("   ├─ Overall repetition rate: {:.1}%", self.repetition_analysis.overall_repetition_rate);
        println!("   ├─ Most repetitive context: {} ({:.1}%)",
                 self.repetition_analysis.most_repetitive_context.unwrap_or(-1),
                 self.repetition_analysis.context_repetition_scores.first()
                     .map(|s| s.repetition_rate).unwrap_or(0.0));
        println!("   └─ Least repetitive context: {} ({:.1}%)",
                 self.repetition_analysis.least_repetitive_context.unwrap_or(-1),
                 self.repetition_analysis.context_repetition_scores.last()
                     .map(|s| s.repetition_rate).unwrap_or(0.0));

        println!("\n📊 STATISTICAL TEST");
        println!("   {}", self.repetition_analysis.statistical_summary.interpretation);

        println!("\n📊 VOCABULARY COMPLEXITY (Top 5 by entropy)");
        println!("   {:<10} {:>10} {:>12} {:>12}",
                 "Context", "Calls", "Mean Entropy", "% Complex");
        println!("   {}", "-".repeat(48));

        for ranking in self.vocabulary_complexity.context_entropy_ranking.iter().take(5) {
            println!("   {:<10} {:>10} {:>12.4} {:>11.1}%",
                     ranking.context_id, ranking.n_calls, ranking.mean_entropy, ranking.pct_complex);
        }

        println!("\n📊 DURATION BY CONTEXT (Top 5 longest)");
        println!("   {:<10} {:>10} {:>12} {:>12}",
                 "Context", "Calls", "Mean (ms)", "Std (ms)");
        println!("   {}", "-".repeat(48));

        for dur in self.duration_analysis.context_durations.iter().take(5) {
            println!("   {:<10} {:>10} {:>12.0} {:>12.0}",
                     dur.context_id, dur.n_calls, dur.mean_ms, dur.std_ms);
        }

        println!("\n📊 KEY FINDINGS");
        for (i, finding) in self.key_findings.iter().enumerate() {
            println!("\n   {}. {}", i + 1, finding);
        }

        // Motif patterns by context
        println!("\n📊 TOP REPETITION PATTERNS BY CONTEXT");
        for ctx in self.context_statistics.iter().take(5) {
            let mut patterns: Vec<_> = ctx.repetition_patterns.iter().collect();
            patterns.sort_by(|a, b| b.1.cmp(a.1));

            println!("\n   Context {} (Top 5 patterns):", ctx.context_id);
            for (pattern, count) in patterns.iter().take(5) {
                println!("      • [{}]: {} occurrences", pattern, count);
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Within-Call Patterns × Behavioral Contexts Combined Analysis  ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  🔗 Combining:                                                  ║");
    println!("║     • Within-call phrase sequences (local types)               ║");
    println!("║     • Behavioral context annotations                           ║");
    println!("║                                                                 ║");
    println!("║  📊 Analyzing:                                                  ║");
    println!("║     • Context vs repetition patterns                           ║");
    println!("║     • Vocabulary complexity by context                         ║");
    println!("║     • Duration and phrase rate by context                      ║");
    println!("║                                                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let annotations_path = data_dir.join("annotations.csv");
    let within_call_path = data_dir.join("within_call_phrase_results/bat_within_call_analyses.json");
    let output_dir = data_dir.join("within_call_context_analysis");

    std::fs::create_dir_all(&output_dir)?;

    // Load data
    println!("📂 Loading annotations from: {}", annotations_path.display());
    let annotations = load_annotations(&annotations_path)?;
    println!("   Loaded {} annotations", annotations.len());

    println!("\n📂 Loading within-call results from: {}", within_call_path.display());
    let analyses = load_within_call_results(&within_call_path)?;
    println!("   Loaded {} within-call analyses", analyses.len());
    println!();

    // Run combined analysis
    println!("🔬 Running combined analysis...");
    let results = analyze_combined(&analyses, &annotations);

    // Print summary
    results.print_summary();

    // Save results
    let results_path = output_dir.join("combined_analysis_results.json");
    let file = File::create(&results_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &results)?;
    println!("\n💾 Results saved to: {}", results_path.display());

    // Save context statistics as CSV
    let csv_path = output_dir.join("context_statistics.csv");
    let mut wtr = csv::Writer::from_writer(BufWriter::new(File::create(&csv_path)?));

    wtr.write_record(&[
        "context_id", "n_vocalizations", "n_phrases", "n_motifs",
        "avg_phrases_per_call", "avg_duration_ms", "avg_entropy",
        "avg_phrase_rate", "pct_with_motifs"
    ])?;

    for ctx in &results.context_statistics {
        wtr.write_record(&[
            ctx.context_id.to_string(),
            ctx.n_vocalizations.to_string(),
            ctx.n_phrases.to_string(),
            ctx.n_motifs.to_string(),
            format!("{:.2}", ctx.avg_phrases_per_call),
            format!("{:.1}", ctx.avg_duration_ms),
            format!("{:.4}", ctx.avg_entropy),
            format!("{:.2}", ctx.avg_phrase_rate),
            format!("{:.1}", ctx.pct_with_motifs),
        ])?;
    }
    wtr.flush()?;
    println!("💾 CSV saved to: {}", csv_path.display());

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}
