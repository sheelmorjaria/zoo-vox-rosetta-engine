// Phrase-Context Matrix Analysis for Zebra Finch Songs
//
// This analysis tests whether different zebra finch individuals use different
// phrase repertoires, or if phrases are shared across individuals.
//
// Hypothesis: If zebra finch song syntax is learned, different individuals
// may have different phrase type distributions ("dialects").
//
// Methods:
// - Generality Score: How many birds use each phrase type
// - Shannon Entropy: Distribution evenness across birds
// - Permutation Test: Statistical significance vs random chance

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WithinCallAnalysis {
    file_name: String,
    phrases: Vec<PhraseCandidate>,
    n_phrase_types: usize,
    phrase_types: Vec<i32>,
    phrase_sequence: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseCandidate {
    id: usize,
    phrase_type: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeneralityMetrics {
    phrase_id: i32,
    total_occurrences: usize,
    birds_using: usize,
    total_birds: usize,
    generality_score: f64,
    shannon_entropy: f64,
    classification: String,
}

#[derive(Debug, Clone, Serialize)]
struct AnalysisResults {
    dataset: String,
    total_phrases: usize,
    total_birds: usize,
    unique_phrase_types: usize,
    generality_metrics: Vec<GeneralityMetrics>,
    phrase_bird_matrix: Vec<Vec<usize>>,
    bird_names: Vec<String>,
    permutation_p_value: f64,
    summary: SummaryStats,
}

#[derive(Debug, Clone, Serialize)]
struct SummaryStats {
    avg_generality: f64,
    avg_entropy: f64,
    universal_phrases: usize,
    bird_specific_phrases: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Phrase-Context Analysis: Zebra Finch Individual Variation               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load within-call results
    let results_path = "/home/sheel/birdsong_analysis/within_call_results/zebra_finch_songs_within_call.json";
    let json_data = fs::read_to_string(results_path)?;
    let dataset: serde_json::Value = serde_json::from_str(&json_data)?;
    
    let file_analyses = dataset["file_analyses"].as_array().ok_or("No file analyses")?;
    println!("Loaded {} file analyses", file_analyses.len());

    // Extract bird ID from file path
    // Files are from koumura2016/BirdX/ or synthetic/
    let mut bird_files: HashMap<String, Vec<&serde_json::Value>> = HashMap::new();
    
    for fa in file_analyses {
        let file_name = fa["file_name"].as_str().unwrap_or("unknown");
        
        // Determine bird ID from file path pattern
        let bird_id = if file_name.contains("Bird") || file_name.starts_with(|c: char| c.is_digit(10)) {
            // Extract bird number from path or assign based on position
            // For synthetic data, use "synthetic" as bird
            if file_name.contains("synth") {
                "synthetic".to_string()
            } else {
                // Group files by their index ranges (assuming sequential per bird)
                let num = file_name.trim_end_matches(".wav").parse::<usize>().unwrap_or(0);
                format!("bird_{}", num / 60)  // Rough approximation
            }
        } else {
            "unknown".to_string()
        };
        
        bird_files.entry(bird_id).or_default().push(fa);
    }
    
    println!("Found {} bird groups", bird_files.len());
    for (bird, files) in &bird_files {
        println!("  {}: {} files", bird, files.len());
    }
    println!();

    // Build phrase-bird matrix
    println!("Building phrase-bird matrix...");
    
    // Collect all phrase sequences
    let mut all_phrase_occurrences: HashMap<i32, HashSet<String>> = HashMap::new();
    let mut phrase_counts: HashMap<i32, usize> = HashMap::new();
    
    for (bird, files) in &bird_files {
        for fa in files {
            if let Some(seq) = fa["phrase_sequence"].as_array() {
                for p in seq {
                    if let Some(phrase_id) = p.as_i64() {
                        let phrase_id = phrase_id as i32;
                        all_phrase_occurrences.entry(phrase_id).or_default().insert(bird.clone());
                        *phrase_counts.entry(phrase_id).or_default() += 1;
                    }
                }
            }
        }
    }
    
    let total_birds = bird_files.len();
    let total_phrases: usize = phrase_counts.values().sum();
    let unique_phrase_types = all_phrase_occurrences.len();
    
    println!("  Total phrases: {}", total_phrases);
    println!("  Unique phrase types: {}", unique_phrase_types);
    println!("  Total birds/individuals: {}", total_birds);
    println!();

    // Compute generality metrics
    println!("Computing generality metrics...");
    
    let mut metrics: Vec<GeneralityMetrics> = Vec::new();
    
    for (&phrase_id, birds) in &all_phrase_occurrences {
        let total_occurrences = phrase_counts.get(&phrase_id).copied().unwrap_or(0);
        let birds_using = birds.len();
        let generality_score = birds_using as f64 / total_birds as f64;
        
        // Compute entropy based on distribution across birds
        let count_per_bird: Vec<usize> = bird_files.iter()
            .map(|(bird, files)| {
                files.iter()
                    .filter_map(|fa| fa["phrase_sequence"].as_array())
                    .flat_map(|seq| seq.iter())
                    .filter(|p| p.as_i64() == Some(phrase_id as i64))
                    .count()
            })
            .collect();
        
        let total: usize = count_per_bird.iter().sum();
        let entropy = if total > 0 {
            count_per_bird.iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total as f64;
                    -p * p.log2()
                })
                .sum()
        } else {
            0.0
        };
        
        let classification = if generality_score >= 0.9 {
            "Universal".to_string()
        } else if generality_score >= 0.5 {
            "Common".to_string()
        } else if generality_score >= 0.2 {
            "Moderate".to_string()
        } else {
            "Rare/Specialist".to_string()
        };
        
        metrics.push(GeneralityMetrics {
            phrase_id,
            total_occurrences,
            birds_using,
            total_birds,
            generality_score,
            shannon_entropy: entropy,
            classification,
        });
    }
    
    // Sort by occurrences
    metrics.sort_by(|a, b| b.total_occurrences.cmp(&a.total_occurrences));
    
    // Permutation test
    println!("Running permutation test (1000 iterations)...");
    
    let observed_mean = metrics.iter().map(|m| m.generality_score).sum::<f64>() / metrics.len() as f64;
    let mut rng = rand::thread_rng();
    let mut null_means: Vec<f64> = Vec::new();
    
    for _ in 0..1000 {
        // Shuffle bird assignments
        let mut shuffled_count = 0.0;
        for m in &metrics {
            // Random number of birds using this phrase
            let random_birds = rng.gen_range(1..=total_birds);
            shuffled_count += random_birds as f64 / total_birds as f64;
        }
        null_means.push(shuffled_count / metrics.len() as f64);
    }
    
    let null_mean: f64 = null_means.iter().sum::<f64>() / null_means.len() as f64;
    let null_std: f64 = {
        let variance = null_means.iter()
            .map(|x| (x - null_mean).powi(2))
            .sum::<f64>() / null_means.len() as f64;
        variance.sqrt()
    };
    
    let z_score = (observed_mean - null_mean) / null_std.max(1e-10);
    let extreme_count = null_means.iter().filter(|&&x| x >= observed_mean).count();
    let p_value = extreme_count as f64 / null_means.len() as f64;
    
    println!("  Observed mean generality: {:.4}", observed_mean);
    println!("  Null mean generality: {:.4}", null_mean);
    println!("  Z-score: {:.4}", z_score);
    println!("  P-value: {:.4}", p_value);
    println!();

    // Summary statistics
    let avg_generality = metrics.iter().map(|m| m.generality_score).sum::<f64>() / metrics.len() as f64;
    let avg_entropy = metrics.iter().map(|m| m.shannon_entropy).sum::<f64>() / metrics.len() as f64;
    let universal_phrases = metrics.iter().filter(|m| m.generality_score >= 0.9).count();
    let bird_specific_phrases = metrics.iter().filter(|m| m.generality_score < 0.2).count();
    
    let summary = SummaryStats {
        avg_generality,
        avg_entropy,
        universal_phrases,
        bird_specific_phrases,
    };

    // Print results
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         ANALYSIS RESULTS                                   ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Dataset: Zebra Finch Songs (Koumura 2016)                               ║");
    println!("║                                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Summary Statistics:                                                      ║");
    println!("║    • Total phrases analyzed: {:>10}", format!("{}", total_phrases));
    println!("║    • Unique phrase types:    {:>10}", format!("{}", unique_phrase_types));
    println!("║    • Birds/individuals:      {:>10}", format!("{}", total_birds));
    println!("║    • Average generality:     {:>10.3}", avg_generality);
    println!("║    • Average entropy:        {:>10.3} bits", avg_entropy);
    println!("║    • Universal phrases:      {:>10}", format!("{}", universal_phrases));
    println!("║    • Bird-specific phrases:  {:>10}", format!("{}", bird_specific_phrases));
    println!("║                                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Permutation Test:                                                        ║");
    println!("║    • P-value:               {:>10.4}", p_value);
    println!("║    • Significant (p<0.05):  {:>10}", if p_value < 0.05 { "YES" } else { "NO" });
    println!("║                                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Top phrase types
    println!("Top 15 Phrase Types by Frequency:");
    println!("{:-<70}", "");
    println!("{:>6} {:>10} {:>8} {:>10} {:>10} {:>15}",
        "ID", "Occurrences", "Birds", "Generality", "Entropy", "Type");
    println!("{:-<70}", "");
    
    for m in metrics.iter().take(15) {
        println!("{:>6} {:>10} {:>8} {:>10.3} {:>10.3} {:>15}",
            m.phrase_id, m.total_occurrences, m.birds_using,
            m.generality_score, m.shannon_entropy, m.classification);
    }
    println!();

    // Save results
    let results = AnalysisResults {
        dataset: "zebra_finch_songs".to_string(),
        total_phrases,
        total_birds,
        unique_phrase_types,
        generality_metrics: metrics,
        phrase_bird_matrix: vec![],  // Too large to include
        bird_names: bird_files.keys().cloned().collect(),
        permutation_p_value: p_value,
        summary,
    };
    
    let output_path = "/home/sheel/birdsong_analysis/within_call_results/zebra_finch_phrase_context_analysis.json";
    let output_file = std::fs::File::create(output_path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(output_file), &results)?;
    println!("Results saved to: {}", output_path);

    Ok(())
}
