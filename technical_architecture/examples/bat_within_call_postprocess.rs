// Post-Processing Analysis for Bat Within-Call Phrase Discovery Results
//
// Analyzes aggregate statistics and consolidates phrase types

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallAnalysisFile {
    pub file_name: String,
    pub call_type: Option<String>,
    pub total_duration_ms: f64,
    pub phrases: Vec<PhraseCandidate>,
    pub n_phrase_types: usize,
    pub phrase_types: Vec<i32>,
    pub motifs: Vec<Motif>,
    pub stats: WithinCallStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidate {
    pub id: usize,
    pub start_ms: f64,
    pub end_ms: f64,
    pub duration_ms: f64,
    pub features: Vec<f64>,
    pub phrase_type: Option<i32>,
    pub type_confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    pub id: usize,
    pub pattern: Vec<i32>,
    pub occurrences: usize,
    pub positions: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallStats {
    pub n_phrases: usize,
    pub avg_phrase_duration_ms: f64,
    pub type_distribution: HashMap<i32, usize>,
    pub type_entropy: f64,
    pub phrase_rate: f64,
    pub avg_within_type_similarity: f64,
    pub avg_between_type_distance: f64,
}

// =============================================================================
// Aggregate Analysis
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct AggregateAnalysis {
    pub total_vocalizations: usize,
    pub total_phrases: usize,
    pub avg_phrases_per_call: f64,
    pub phrase_duration_stats: DurationStats,
    pub top_motifs: Vec<MotifSummary>,
    pub motif_length_distribution: HashMap<usize, usize>,
    pub phrases_per_call_distribution: HashMap<usize, usize>,
    pub entropy_stats: EntropyStats,
    pub files_with_motifs: usize,
    pub pct_with_motifs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DurationStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub std_ms: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntropyStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub simple_count: usize,  // E < 0.5
    pub medium_count: usize,  // 0.5 <= E < 1.5
    pub complex_count: usize, // E >= 1.5
}

#[derive(Debug, Clone, Serialize)]
pub struct MotifSummary {
    pub pattern: Vec<i32>,
    pub total_occurrences: usize,
    pub n_vocalizations: usize,
    pub example_files: Vec<String>,
}

struct MotifAggregator {
    pattern: Vec<i32>,
    total_occurrences: usize,
    n_vocalizations: usize,
    example_files: Vec<String>,
}

// =============================================================================
// Analysis Functions
// =============================================================================

fn analyze_aggregate(analyses: &[WithinCallAnalysisFile]) -> AggregateAnalysis {
    let total_vocalizations = analyses.len();

    // Collect statistics
    let mut all_durations: Vec<f64> = Vec::new();
    let mut all_entropies: Vec<f64> = Vec::new();
    let mut phrases_per_call: HashMap<usize, usize> = HashMap::new();
    let mut motif_patterns: HashMap<Vec<i32>, MotifAggregator> = HashMap::new();
    let mut motif_lengths: HashMap<usize, usize> = HashMap::new();
    let mut files_with_motifs = 0;
    let mut total_phrases = 0;

    for analysis in analyses {
        total_phrases += analysis.stats.n_phrases;

        *phrases_per_call.entry(analysis.stats.n_phrases).or_default() += 1;

        for phrase in &analysis.phrases {
            all_durations.push(phrase.duration_ms);
        }

        if analysis.stats.n_phrases > 1 {
            all_entropies.push(analysis.stats.type_entropy);
        }

        if !analysis.motifs.is_empty() {
            files_with_motifs += 1;
        }

        for motif in &analysis.motifs {
            *motif_lengths.entry(motif.pattern.len()).or_default() += motif.occurrences;

            let aggregator = motif_patterns
                .entry(motif.pattern.clone())
                .or_insert_with(|| MotifAggregator {
                    pattern: motif.pattern.clone(),
                    total_occurrences: 0,
                    n_vocalizations: 0,
                    example_files: Vec::new(),
                });

            aggregator.total_occurrences += motif.occurrences;
            aggregator.n_vocalizations += 1;
            if aggregator.example_files.len() < 3 {
                aggregator.example_files.push(analysis.file_name.clone());
            }
        }
    }

    // Duration stats
    let phrase_duration_stats = compute_duration_stats(&all_durations);

    // Entropy stats
    let entropy_stats = compute_entropy_stats(&all_entropies);

    // Top motifs
    let mut top_motifs: Vec<MotifSummary> = motif_patterns
        .into_values()
        .map(|agg| MotifSummary {
            pattern: agg.pattern,
            total_occurrences: agg.total_occurrences,
            n_vocalizations: agg.n_vocalizations,
            example_files: agg.example_files,
        })
        .collect();

    top_motifs.sort_by(|a, b| b.total_occurrences.cmp(&a.total_occurrences));
    top_motifs.truncate(50);

    AggregateAnalysis {
        total_vocalizations,
        total_phrases,
        avg_phrases_per_call: total_phrases as f64 / total_vocalizations as f64,
        phrase_duration_stats,
        top_motifs,
        motif_length_distribution: motif_lengths,
        phrases_per_call_distribution: phrases_per_call,
        entropy_stats,
        files_with_motifs,
        pct_with_motifs: files_with_motifs as f64 / total_vocalizations as f64 * 100.0,
    }
}

fn compute_duration_stats(durations: &[f64]) -> DurationStats {
    if durations.is_empty() {
        return DurationStats {
            min_ms: 0.0,
            max_ms: 0.0,
            mean_ms: 0.0,
            median_ms: 0.0,
            std_ms: 0.0,
            p10: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p90: 0.0,
        };
    }

    let mut sorted = durations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    DurationStats {
        min_ms: sorted[0],
        max_ms: sorted[n - 1],
        mean_ms: mean,
        median_ms: sorted[n / 2],
        std_ms: variance.sqrt(),
        p10: sorted[(n as f64 * 0.10) as usize],
        p25: sorted[(n as f64 * 0.25) as usize],
        p50: sorted[n / 2],
        p75: sorted[(n as f64 * 0.75) as usize],
        p90: sorted[(n as f64 * 0.90) as usize],
    }
}

fn compute_entropy_stats(entropies: &[f64]) -> EntropyStats {
    if entropies.is_empty() {
        return EntropyStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            simple_count: 0,
            medium_count: 0,
            complex_count: 0,
        };
    }

    EntropyStats {
        min: entropies.iter().cloned().fold(f64::INFINITY, f64::min),
        max: entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mean: entropies.iter().sum::<f64>() / entropies.len() as f64,
        simple_count: entropies.iter().filter(|&&e| e < 0.5).count(),
        medium_count: entropies.iter().filter(|&&e| e >= 0.5 && e < 1.5).count(),
        complex_count: entropies.iter().filter(|&&e| e >= 1.5).count(),
    }
}

// =============================================================================
// Phrase Type Consolidation
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct GlobalVocabulary {
    pub n_global_types: usize,
    pub consolidation_ratio: f64,
    pub type_frequencies: Vec<(i32, usize)>,
}

fn consolidate_phrase_types(analyses: &[WithinCallAnalysisFile], similarity_threshold: f64) -> GlobalVocabulary {
    // Collect all phrase type representatives
    let mut local_types: Vec<(String, i32, Vec<f64>)> = Vec::new();

    for analysis in analyses {
        let mut type_features: HashMap<i32, Vec<Vec<f64>>> = HashMap::new();

        for phrase in &analysis.phrases {
            if let Some(local_type) = phrase.phrase_type {
                type_features
                    .entry(local_type)
                    .or_default()
                    .push(phrase.features.clone());
            }
        }

        for (local_type, features_list) in type_features {
            let n_features = features_list[0].len();
            let mut mean_features = vec![0.0; n_features];

            for features in &features_list {
                for (i, &f) in features.iter().enumerate() {
                    mean_features[i] += f;
                }
            }

            for f in &mut mean_features {
                *f /= features_list.len() as f64;
            }

            local_types.push((analysis.file_name.clone(), local_type, mean_features));
        }
    }

    // Consolidate into global types
    let mut global_types: Vec<Vec<f64>> = Vec::new();
    let mut type_counts: Vec<usize> = Vec::new();
    let feature_dim = local_types.first().map(|(_, _, f)| f.len()).unwrap_or(30);

    for (_, _, features) in &local_types {
        let query = Array1::from_vec(features.clone());

        // Find best matching global type
        let mut best_match: Option<(usize, f64)> = None;

        for (i, global_features) in global_types.iter().enumerate() {
            let global_feat = Array1::from_vec(global_features.clone());
            let dist: f64 = query.iter().zip(global_feat.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            let sim = 1.0 - (-dist.sqrt()).exp();

            if sim >= similarity_threshold {
                if best_match.is_none() || sim > best_match.unwrap().1 {
                    best_match = Some((i, sim));
                }
            }
        }

        match best_match {
            Some((idx, _)) => {
                type_counts[idx] += 1;
            }
            None => {
                global_types.push(features.clone());
                type_counts.push(1);
            }
        }
    }

    let n_global_types = global_types.len();
    let consolidation_ratio = local_types.len() as f64 / n_global_types as f64;

    // Get top types by frequency
    let mut type_frequencies: Vec<(i32, usize)> = type_counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| (i as i32, count))
        .collect();
    type_frequencies.sort_by(|a, b| b.1.cmp(&a.1));
    type_frequencies.truncate(20);

    GlobalVocabulary {
        n_global_types,
        consolidation_ratio,
        type_frequencies,
    }
}

// =============================================================================
// Output
// =============================================================================

impl AggregateAnalysis {
    fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           Aggregate Within-Call Analysis Summary               ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📊 OVERVIEW");
        println!("   ├─ Total vocalizations: {:>12}", self.total_vocalizations);
        println!("   ├─ Total phrases:       {:>12}", self.total_phrases);
        println!("   └─ Avg phrases/call:    {:>12.2}", self.avg_phrases_per_call);

        println!("\n📊 PHRASE DURATION (ms)");
        println!("   ├─ Min:    {:>8.1}", self.phrase_duration_stats.min_ms);
        println!("   ├─ Max:    {:>8.1}", self.phrase_duration_stats.max_ms);
        println!("   ├─ Mean:   {:>8.1}", self.phrase_duration_stats.mean_ms);
        println!("   ├─ Median: {:>8.1}", self.phrase_duration_stats.median_ms);
        println!("   ├─ Std:    {:>8.1}", self.phrase_duration_stats.std_ms);
        println!("   ├─ P10:    {:>8.1}", self.phrase_duration_stats.p10);
        println!("   ├─ P25:    {:>8.1}", self.phrase_duration_stats.p25);
        println!("   ├─ P75:    {:>8.1}", self.phrase_duration_stats.p75);
        println!("   └─ P90:    {:>8.1}", self.phrase_duration_stats.p90);

        println!("\n📊 MOTIF STATISTICS");
        println!(
            "   ├─ Files with motifs: {} ({:.1}%)",
            self.files_with_motifs, self.pct_with_motifs
        );

        let total_motif_occurrences: usize = self.motif_length_distribution.values().sum();
        println!("   ├─ Total motif occurrences: {}", total_motif_occurrences);

        let mut lengths: Vec<_> = self.motif_length_distribution.iter().collect();
        lengths.sort_by_key(|(k, _)| *k);
        for (len, count) in lengths.iter().take(5) {
            println!("   • Motif length {}: {} occurrences", len, count);
        }

        println!("\n📊 TOP MOTIF PATTERNS (across all vocalizations)");
        for (i, motif) in self.top_motifs.iter().take(10).enumerate() {
            let pattern_str: String = motif
                .pattern
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join("-");
            println!(
                "   {}. [{}] : {} occurrences in {} files",
                i + 1,
                pattern_str,
                motif.total_occurrences,
                motif.n_vocalizations
            );
        }

        println!("\n📊 VOCABULARY COMPLEXITY (Type Entropy)");
        println!("   ├─ Min entropy: {:.3}", self.entropy_stats.min);
        println!("   ├─ Max entropy: {:.3}", self.entropy_stats.max);
        println!("   ├─ Mean entropy: {:.3}", self.entropy_stats.mean);
        let total_with_phrases =
            self.entropy_stats.simple_count + self.entropy_stats.medium_count + self.entropy_stats.complex_count;
        println!(
            "   ├─ Simple (E<0.5): {} ({:.1}%)",
            self.entropy_stats.simple_count,
            self.entropy_stats.simple_count as f64 / total_with_phrases as f64 * 100.0
        );
        println!(
            "   ├─ Medium (0.5≤E<1.5): {} ({:.1}%)",
            self.entropy_stats.medium_count,
            self.entropy_stats.medium_count as f64 / total_with_phrases as f64 * 100.0
        );
        println!(
            "   └─ Complex (E≥1.5): {} ({:.1}%)",
            self.entropy_stats.complex_count,
            self.entropy_stats.complex_count as f64 / total_with_phrases as f64 * 100.0
        );

        println!("\n📊 PHRASES PER VOCALIZATION");
        let mut ppc: Vec<_> = self.phrases_per_call_distribution.iter().collect();
        ppc.sort_by_key(|(k, _)| *k);
        for (n_phrases, count) in ppc.iter().take(10) {
            let pct = **count as f64 / self.total_vocalizations as f64 * 100.0;
            println!("   • {} phrases: {} calls ({:.1}%)", n_phrases, count, pct);
        }
        if ppc.len() > 10 {
            let remaining: usize = ppc.iter().skip(10).map(|(_, c)| *c).sum();
            println!("   ... and {} more calls with >10 phrases", remaining);
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║    Bat Within-Call Results: Post-Processing Analysis          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let results_path = PathBuf::from(
        "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/within_call_phrase_results/bat_within_call_analyses.json",
    );
    let output_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/within_call_phrase_results");

    println!("\n📂 Loading analysis results from: {}", results_path.display());

    let file = File::open(&results_path)?;
    let analyses: Vec<WithinCallAnalysisFile> = serde_json::from_reader(BufReader::new(file))?;

    println!("   Loaded {} vocalization analyses", analyses.len());

    // Aggregate analysis
    let aggregate = analyze_aggregate(&analyses);
    aggregate.print_summary();

    // Save aggregate results
    let aggregate_path = output_dir.join("aggregate_analysis.json");
    let file = File::create(&aggregate_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &aggregate)?;
    println!("\n💾 Saved aggregate analysis to: {}", aggregate_path.display());

    // Consolidate phrase types with different thresholds
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║    Phrase Type Consolidation (Similarity Threshold Testing)   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let mut consolidation_results: Vec<(f64, GlobalVocabulary)> = Vec::new();

    for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9] {
        let vocab = consolidate_phrase_types(&analyses, threshold);
        println!(
            "   Threshold {:.2}: {:>6} global types ({:.1}x consolidation)",
            threshold, vocab.n_global_types, vocab.consolidation_ratio
        );
        consolidation_results.push((threshold, vocab));
    }

    // Save consolidation results
    let consolidation_path = output_dir.join("consolidation_analysis.json");
    let consolidation_data: Vec<_> = consolidation_results
        .into_iter()
        .map(|(threshold, vocab)| {
            serde_json::json!({
                "threshold": threshold,
                "n_global_types": vocab.n_global_types,
                "consolidation_ratio": vocab.consolidation_ratio,
                "top_types": vocab.type_frequencies,
            })
        })
        .collect();

    let file = File::create(&consolidation_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &consolidation_data)?;
    println!("\n💾 Saved consolidation analysis to: {}", consolidation_path.display());

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      ANALYSIS COMPLETE                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}
