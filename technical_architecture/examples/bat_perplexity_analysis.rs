//! Bat Perplexity Analysis with Syntactic Depth 6
//!
//! This example analyzes bat corpus analysis results, computing perplexity,
//! entropy, and ZIPF correlation metrics to evaluate language-like structure.

use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};

/// Corpus analysis report structure (from bat_corpus_analysis_report.json)
#[derive(Debug, Deserialize, Serialize)]
pub struct CorpusAnalysisReport {
    pub config: CorpusConfig,
    pub total_vocalizations: usize,
    pub total_segments: usize,
    pub unique_segment_types: usize,
    pub unique_ngrams: usize,
    pub max_ngram_length: usize,
    pub avg_segments_per_vocalization: f64,
    pub top_bigrams: Vec<(Vec<i32>, usize)>,
    pub top_trigrams: Vec<(Vec<i32>, usize)>,
    pub top_4grams: Vec<(Vec<i32>, usize)>,
    pub top_5grams: Vec<(Vec<i32>, usize)>,
    pub longest_repeated_ngram: (Vec<i32>, usize),
    pub analysis_timestamp: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CorpusConfig {
    pub vocabulary_size: usize,
    pub max_ngram_length: usize,
}

/// Perplexity analysis results
#[derive(Debug, Serialize)]
pub struct PerplexityAnalysis {
    pub corpus_stats: CorpusStats,
    pub ngram_analysis: Vec<NgramAnalysis>,
    pub zipf_analysis: ZipfAnalysis,
    pub language_structure_score: f64,
}

#[derive(Debug, Serialize)]
pub struct CorpusStats {
    pub total_vocalizations: usize,
    pub total_segments: usize,
    pub vocabulary_size: usize,
    pub avg_segments_per_vocalization: f64,
    pub type_token_ratio: f64,
}

#[derive(Debug, Serialize)]
pub struct NgramAnalysis {
    pub order: usize,
    pub unique_count: usize,
    pub total_count: usize,
    pub entropy: f64,
    pub perplexity: f64,
    pub most_frequent: Vec<i32>,
    pub most_frequent_count: usize,
    pub coverage: f64,
}

#[derive(Debug, Serialize)]
pub struct ZipfAnalysis {
    pub correlation: f64,
    pub exponent: f64,
    pub interpretation: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     Bat Perplexity Analysis (Syntactic Depth 6)                         ║");
    println!("║     Corpus Analysis Results → Language Structure Metrics                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load corpus analysis report
    let report_path = "bat_corpus_analysis_report.json";
    println!("Loading corpus analysis from: {}", report_path);

    let content = fs::read_to_string(report_path)?;
    let report: CorpusAnalysisReport = serde_json::from_str(&content)?;

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  [1] Corpus Statistics                                                   ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("  Total Vocalizations:     {:>12}", report.total_vocalizations);
    println!("  Total Segments:          {:>12}", report.total_segments);
    println!("  Unique Segment Types:    {:>12}", report.unique_segment_types);
    println!("  Vocabulary Size:         {:>12}", report.config.vocabulary_size);
    println!("  Avg Segments/Vocalization: {:>10.2}", report.avg_segments_per_vocalization);
    println!("  Unique N-grams:          {:>12}", report.unique_ngrams);
    println!("  Max N-gram Length:       {:>12}", report.max_ngram_length);
    println!();

    // Type-Token Ratio (measure of vocabulary diversity)
    let ttr = report.unique_segment_types as f64 / report.total_segments as f64;
    println!("  Type-Token Ratio:        {:>12.4}", ttr);
    if ttr < 0.1 {
        println!("    → Low diversity (many repetitions of few types)");
    } else if ttr < 0.3 {
        println!("    → Moderate diversity (typical for animal communication)");
    } else {
        println!("    → High diversity (large vocabulary usage)");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // N-gram Perplexity Analysis
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  [2] N-gram Perplexity Analysis (Orders 2-6)                            ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Compute entropy and perplexity for each n-gram order
    let ngram_data = vec![
        (2, &report.top_bigrams),
        (3, &report.top_trigrams),
        (4, &report.top_4grams),
        (5, &report.top_5grams),
    ];

    println!("┌─────────┬────────────┬────────────┬────────────┬────────────┬────────────┐");
    println!("│ Order   │ Unique     │ Top Freq   │ Entropy    │ Perplexity │ Coverage   │");
    println!("├─────────┼────────────┼────────────┼────────────┼────────────┼────────────┤");

    let mut ngram_analyses: Vec<NgramAnalysis> = Vec::new();
    let vocabulary_size = report.config.vocabulary_size as f64;

    for (order, ngrams) in &ngram_data {
        let unique_count = ngrams.len();
        let total_count: usize = ngrams.iter().map(|(_, c)| *c).sum();

        // Compute frequency distribution
        let freqs: Vec<f64> = ngrams.iter().map(|(_, c)| *c as f64 / total_count as f64).collect();

        // Compute entropy
        let entropy: f64 = freqs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        // Perplexity = 2^entropy
        let perplexity = 2.0_f64.powf(entropy);

        // Coverage = how many possible n-grams are observed
        let possible_ngrams = vocabulary_size.powi(*order as i32);
        let coverage = unique_count as f64 / possible_ngrams;

        // Most frequent
        let (most_frequent, most_frequent_count) = ngrams.first()
            .map(|(ngram, count)| (ngram.clone(), *count))
            .unwrap_or((vec![], 0));

        println!(
            "│ {:<7} │ {:<10} │ {:<10} │ {:<10.4} │ {:<10.4} │ {:<10.6} │",
            format!("{}-gram", order),
            unique_count,
            most_frequent_count,
            entropy,
            perplexity,
            coverage
        );

        ngram_analyses.push(NgramAnalysis {
            order: *order,
            unique_count,
            total_count,
            entropy,
            perplexity,
            most_frequent,
            most_frequent_count,
            coverage,
        });
    }
    println!("└─────────┴────────────┴────────────┴────────────┴────────────┴────────────┘");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // ZIPF Distribution Analysis
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  [3] ZIPF Distribution Analysis                                         ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Extract bigram frequencies for ZIPF analysis
    let mut frequencies: Vec<usize> = report.top_bigrams.iter().map(|(_, c)| *c).collect();
    frequencies.sort_by(|a, b| b.cmp(a)); // Sort descending

    // Compute ZIPF correlation (log-log regression)
    let zipf_correlation = compute_zipf_correlation(&frequencies);

    // Estimate ZIPF exponent (slope of log-log plot)
    let zipf_exponent = estimate_zipf_exponent(&frequencies);

    println!("  Bigram Frequency Distribution:");
    println!("    Rank 1:   {} occurrences", frequencies.first().unwrap_or(&0));
    println!("    Rank 10:  {} occurrences", frequencies.get(9).unwrap_or(&0));
    println!("    Rank 50:  {} occurrences", frequencies.get(49).unwrap_or(&0));
    println!("    Rank 100: {} occurrences", frequencies.get(99).unwrap_or(&0));
    println!();
    println!("  ZIPF Analysis:");
    println!("    Correlation (r):     {:>10.4}", zipf_correlation);
    println!("    Exponent (α):        {:>10.4}", zipf_exponent);
    println!();

    let interpretation = if zipf_correlation > 0.9 {
        "Very strong language-like distribution (r > 0.9)"
    } else if zipf_correlation > 0.8 {
        "Strong language-like distribution (r > 0.8)"
    } else if zipf_correlation > 0.6 {
        "Moderate language-like structure (r > 0.6)"
    } else {
        "Weak language-like structure (r < 0.6)"
    };

    println!("    Interpretation: {}", interpretation);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Longest Repeated N-gram
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  [4] Longest Repeated N-gram                                            ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    let (longest_ngram, longest_count) = &report.longest_repeated_ngram;
    println!("  Pattern Length:  {} elements", longest_ngram.len());
    println!("  Occurrences:     {}", longest_count);
    println!("  Pattern:         {:?}", longest_ngram);
    println!();

    if longest_ngram.len() >= 5 {
        println!("  → Syntactic depth of at least {} detected!", longest_ngram.len());
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Language Structure Score
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  [5] Language Structure Assessment                                      ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Compute composite language structure score
    // Based on: ZIPF correlation, vocabulary diversity, syntactic depth, n-gram structure

    let zipf_score = (zipf_correlation + 1.0) / 2.0; // Normalize to 0-1

    // Vocabulary score: reward moderate TTR (0.05-0.2 typical for structured communication)
    let vocab_score = if ttr >= 0.05 && ttr <= 0.3 {
        1.0 - (ttr - 0.15).abs() / 0.15
    } else {
        0.3
    };

    // Syntactic depth score: based on longest repeated n-gram
    let depth_score = (longest_ngram.len() as f64 / 6.0).min(1.0);

    // N-gram structure score: based on perplexity reduction from unigram
    let unigram_perplexity = vocabulary_size.log2().exp2();
    let bigram_perplexity = ngram_analyses.first()
        .map(|a| a.perplexity)
        .unwrap_or(unigram_perplexity);
    let perplexity_reduction = (unigram_perplexity - bigram_perplexity) / unigram_perplexity;
    let structure_score = perplexity_reduction.max(0.0).min(1.0);

    // Composite score (weighted average)
    let language_score = 0.35 * zipf_score + 0.15 * vocab_score + 0.20 * depth_score + 0.30 * structure_score;

    println!("  Component Scores:");
    println!("    ZIPF Distribution:    {:>6.1}%", zipf_score * 100.0);
    println!("    Vocabulary Diversity: {:>6.1}%", vocab_score * 100.0);
    println!("    Syntactic Depth:      {:>6.1}%", depth_score * 100.0);
    println!("    N-gram Structure:     {:>6.1}%", structure_score * 100.0);
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  COMPOSITE LANGUAGE STRUCTURE SCORE: {:>6.1}%               │", language_score * 100.0);
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // Final interpretation
    if language_score > 0.7 {
        println!("  Assessment: STRONG evidence of combinatorial syntax");
        println!("  → Bat vocalizations exhibit language-like structure");
    } else if language_score > 0.5 {
        println!("  Assessment: MODERATE evidence of combinatorial syntax");
        println!("  → Some language-like patterns detected");
    } else if language_score > 0.3 {
        println!("  Assessment: WEAK evidence of combinatorial syntax");
        println!("  → Limited syntactic structure observed");
    } else {
        println!("  Assessment: MINIMAL evidence of combinatorial syntax");
        println!("  → Primarily fixed or graded signals");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Corpus:           {:>10} vocalizations", report.total_vocalizations);
    println!("  Segments:         {:>10} total", report.total_segments);
    println!("  Vocabulary:       {:>10} types", report.unique_segment_types);
    println!("  N-grams Found:    {:>10} unique", report.unique_ngrams);
    println!("  ZIPF Correlation: {:>10.4}", zipf_correlation);
    println!("  Max Syntactic Depth: {:>7} elements", report.longest_repeated_ngram.0.len());
    println!("  Language Score:   {:>10.1}%", language_score * 100.0);
    println!();

    // Save analysis results
    let analysis = PerplexityAnalysis {
        corpus_stats: CorpusStats {
            total_vocalizations: report.total_vocalizations,
            total_segments: report.total_segments,
            vocabulary_size: report.config.vocabulary_size,
            avg_segments_per_vocalization: report.avg_segments_per_vocalization,
            type_token_ratio: ttr,
        },
        ngram_analysis: ngram_analyses,
        zipf_analysis: ZipfAnalysis {
            correlation: zipf_correlation,
            exponent: zipf_exponent,
            interpretation: interpretation.to_string(),
        },
        language_structure_score: language_score,
    };

    let output_path = "bat_perplexity_analysis_results.json";
    let output = serde_json::to_string_pretty(&analysis)?;
    fs::write(output_path, output)?;
    println!("  Analysis saved to: {}", output_path);

    Ok(())
}

/// Compute ZIPF correlation from sorted frequency counts
fn compute_zipf_correlation(frequencies: &[usize]) -> f64 {
    if frequencies.len() < 10 {
        return 0.0;
    }

    let n = frequencies.len().min(1000) as f64; // Use top 1000 for stability

    // Compute means of log(rank) and log(frequency)
    let sum_log_rank: f64 = (1..=n as usize)
        .map(|r| (r as f64).ln())
        .sum();
    let mean_log_rank = sum_log_rank / n;

    let sum_log_freq: f64 = frequencies
        .iter()
        .take(n as usize)
        .map(|f| (*f as f64).ln())
        .sum();
    let mean_log_freq = sum_log_freq / n;

    // Compute Pearson correlation
    let mut cov = 0.0;
    let mut var_rank = 0.0;
    let mut var_freq = 0.0;

    for (i, freq) in frequencies.iter().take(n as usize).enumerate() {
        let log_rank = ((i + 1) as f64).ln();
        let log_freq = (*freq as f64).ln();

        let diff_rank = log_rank - mean_log_rank;
        let diff_freq = log_freq - mean_log_freq;

        cov += diff_rank * diff_freq;
        var_rank += diff_rank * diff_rank;
        var_freq += diff_freq * diff_freq;
    }

    if var_rank > 0.0 && var_freq > 0.0 {
        -cov / (var_rank.sqrt() * var_freq.sqrt()) // Negative because ZIPF is inverse
    } else {
        0.0
    }
}

/// Estimate ZIPF exponent using linear regression on log-log plot
fn estimate_zipf_exponent(frequencies: &[usize]) -> f64 {
    if frequencies.len() < 10 {
        return 0.0;
    }

    let n = frequencies.len().min(1000);

    // Linear regression: log(frequency) = intercept - exponent * log(rank)
    let sum_x: f64 = (1..=n).map(|r| (r as f64).ln()).sum();
    let sum_y: f64 = frequencies.iter().take(n).map(|f| (*f as f64).ln()).sum();
    let sum_xy: f64 = (1..=n)
        .zip(frequencies.iter().take(n))
        .map(|(r, f)| (r as f64).ln() * (*f as f64).ln())
        .sum();
    let sum_x2: f64 = (1..=n).map(|r| (r as f64).ln().powi(2)).sum();

    let n_f = n as f64;
    let denominator = n_f * sum_x2 - sum_x * sum_x;

    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    // Slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
    // ZIPF exponent = -slope
    let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;
    -slope
}
