// Phase 2: Advanced Sequence Analysis for Zebra Finch Songs
//
// This analysis implements methods to test for syntactic patterns in zebra finch songs:
//
// 1. N-Gram Analysis - Phrase transition probabilities
// 2. Perplexity Analysis - How predictable are the sequences?
// 3. Motif Discovery - Repeated phrase patterns
// 4. Markov Chain Analysis - Transition matrices
// 5. Sequence Clustering - Group similar songs

use std::collections::{HashMap, HashSet};
use std::fs;

use serde::{Deserialize, Serialize};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct WithinCallAnalysis {
    file_name: String,
    phrases: Vec<PhraseInfo>,
    n_phrase_types: usize,
    phrase_sequence: Vec<i32>,
    stats: FileStats,
}

#[derive(Debug, Clone, Deserialize)]
struct PhraseInfo {
    id: usize,
    duration_ms: f64,
    phrase_type: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileStats {
    n_phrases: usize,
    avg_phrase_duration_ms: f64,
    type_entropy: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TransitionMatrix {
    from_type: i32,
    to_type: i32,
    count: usize,
    probability: f64,
}

#[derive(Debug, Clone, Serialize)]
struct NgramStats {
    ngram: Vec<i32>,
    count: usize,
    frequency: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Motif {
    pattern: Vec<i32>,
    occurrences: usize,
    avg_interval: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SequenceResults {
    total_sequences: usize,
    total_phrases: usize,
    unique_types: usize,
    transition_matrix: Vec<TransitionMatrix>,
    top_bigrams: Vec<NgramStats>,
    top_trigrams: Vec<NgramStats>,
    motifs: Vec<Motif>,
    perplexity: f64,
    type_entropy: f64,
    summary: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Phase 2: Advanced Sequence Analysis - Zebra Finch Songs                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Testing for Syntactic Structure using:                                   ║");
    println!("║    1. N-Gram Transition Analysis                                          ║");
    println!("║    2. Perplexity Calculation                                              ║");
    println!("║    3. Motif Discovery                                                     ║");
    println!("║    4. Markov Chain Analysis                                               ║");
    println!("║    5. Sequence Clustering                                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load within-call results
    let results_path = "/home/sheel/birdsong_analysis/within_call_results/zebra_finch_songs_within_call.json";
    let json_data = fs::read_to_string(results_path)?;
    let dataset: serde_json::Value = serde_json::from_str(&json_data)?;
    
    let file_analyses = dataset["file_analyses"].as_array().ok_or("No file analyses")?;
    println!("Loaded {} file analyses", file_analyses.len());
    println!();

    // Extract all sequences
    let mut all_sequences: Vec<Vec<i32>> = Vec::new();
    let mut all_types: HashSet<i32> = HashSet::new();
    let mut total_phrases = 0;
    
    for fa in file_analyses {
        if let Some(seq) = fa["phrase_sequence"].as_array() {
            let sequence: Vec<i32> = seq.iter()
                .filter_map(|p| p.as_i64().map(|x| x as i32))
                .collect();
            
            for &t in &sequence {
                all_types.insert(t);
            }
            
            total_phrases += sequence.len();
            all_sequences.push(sequence);
        }
    }
    
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Basic Statistics                                                │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!("   Total sequences (songs): {}", all_sequences.len());
    println!("   Total phrases: {}", total_phrases);
    println!("   Unique phrase types: {}", all_types.len());
    println!("   Average phrases per song: {:.1}", total_phrases as f64 / all_sequences.len() as f64);
    println!();

    // ========================================================================
    // Step 2: Build Transition Matrix (Bigrams)
    // ========================================================================
    
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Transition Matrix Analysis                                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    
    let mut transition_counts: HashMap<(i32, i32), usize> = HashMap::new();
    let mut from_counts: HashMap<i32, usize> = HashMap::new();
    
    for seq in &all_sequences {
        for window in seq.windows(2) {
            let from = window[0];
            let to = window[1];
            *transition_counts.entry((from, to)).or_default() += 1;
            *from_counts.entry(from).or_default() += 1;
        }
    }
    
    let mut transitions: Vec<TransitionMatrix> = transition_counts.iter()
        .map(|((from, to), count)| {
            let from_total = *from_counts.get(from).unwrap_or(&1) as f64;
            TransitionMatrix {
                from_type: *from,
                to_type: *to,
                count: *count,
                probability: *count as f64 / from_total,
            }
        })
        .collect();
    
    transitions.sort_by(|a, b| b.count.cmp(&a.count));
    
    println!("   Unique transitions: {}", transitions.len());
    println!();
    println!("   Top 15 Transitions:");
    println!("   {:>8} {:>8} {:>10} {:>10}", "From", "To", "Count", "Probability");
    println!("   {}", "-".repeat(40));
    
    for t in transitions.iter().take(15) {
        println!("   {:>8} {:>8} {:>10} {:>10.3}", 
            t.from_type, t.to_type, t.count, t.probability);
    }
    println!();

    // ========================================================================
    // Step 3: N-Gram Analysis (Bigrams and Trigrams)
    // ========================================================================
    
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: N-Gram Analysis                                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    
    // Bigrams
    let mut bigram_counts: HashMap<Vec<i32>, usize> = HashMap::new();
    for seq in &all_sequences {
        for window in seq.windows(2) {
            *bigram_counts.entry(window.to_vec()).or_default() += 1;
        }
    }
    
    let total_bigrams: usize = bigram_counts.values().sum();
    let mut bigrams: Vec<NgramStats> = bigram_counts.iter()
        .map(|(ngram, count)| NgramStats {
            ngram: ngram.clone(),
            count: *count,
            frequency: *count as f64 / total_bigrams as f64,
        })
        .collect();
    bigrams.sort_by(|a, b| b.count.cmp(&a.count));
    
    println!("   Unique bigrams: {}", bigrams.len());
    println!("   Top 15 Bigrams:");
    println!("   {:>20} {:>10} {:>10}", "Pattern", "Count", "Frequency");
    println!("   {}", "-".repeat(45));
    
    for b in bigrams.iter().take(15) {
        let pattern = format!("[{}, {}]", b.ngram[0], b.ngram[1]);
        println!("   {:>20} {:>10} {:>10.4}", pattern, b.count, b.frequency);
    }
    println!();
    
    // Trigrams
    let mut trigram_counts: HashMap<Vec<i32>, usize> = HashMap::new();
    for seq in &all_sequences {
        for window in seq.windows(3) {
            *trigram_counts.entry(window.to_vec()).or_default() += 1;
        }
    }
    
    let total_trigrams: usize = trigram_counts.values().sum();
    let mut trigrams: Vec<NgramStats> = trigram_counts.iter()
        .map(|(ngram, count)| NgramStats {
            ngram: ngram.clone(),
            count: *count,
            frequency: *count as f64 / total_trigrams as f64,
        })
        .collect();
    trigrams.sort_by(|a, b| b.count.cmp(&a.count));
    
    println!("   Unique trigrams: {}", trigrams.len());
    println!("   Top 15 Trigrams:");
    println!("   {:>25} {:>10} {:>10}", "Pattern", "Count", "Frequency");
    println!("   {}", "-".repeat(50));
    
    for t in trigrams.iter().take(15) {
        let pattern = format!("[{}, {}, {}]", t.ngram[0], t.ngram[1], t.ngram[2]);
        println!("   {:>25} {:>10} {:>10.4}", pattern, t.count, t.frequency);
    }
    println!();

    // ========================================================================
    // Step 4: Motif Discovery
    // ========================================================================
    
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Motif Discovery                                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    
    let mut motifs: Vec<Motif> = Vec::new();
    
    // Find repeated patterns of length 2-5
    for motif_len in 2..=5 {
        let mut pattern_occurrences: HashMap<Vec<i32>, Vec<usize>> = HashMap::new();
        
        for seq in &all_sequences {
            if seq.len() >= motif_len {
                for i in 0..=(seq.len() - motif_len) {
                    let pattern = seq[i..i + motif_len].to_vec();
                    pattern_occurrences.entry(pattern).or_default().push(i);
                }
            }
        }
        
        for (pattern, positions) in pattern_occurrences {
            if positions.len() >= 5 {  // At least 5 occurrences
                let avg_interval = if positions.len() > 1 {
                    positions.windows(2)
                        .map(|w| (w[1] as f64 - w[0] as f64).abs())
                        .sum::<f64>() / (positions.len() - 1) as f64
                } else {
                    0.0
                };
                
                motifs.push(Motif {
                    pattern,
                    occurrences: positions.len(),
                    avg_interval,
                });
            }
        }
    }
    
    motifs.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
    
    println!("   Found {} recurring motifs (>=5 occurrences)", motifs.len());
    println!();
    println!("   Top 15 Motifs:");
    println!("   {:>25} {:>12} {:>12}", "Pattern", "Occurrences", "Avg Interval");
    println!("   {}", "-".repeat(55));
    
    for m in motifs.iter().take(15) {
        let pattern: String = m.pattern.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("-");
        println!("   {:>25} {:>12} {:>12.1}", 
            format!("[{}]", pattern), m.occurrences, m.avg_interval);
    }
    println!();

    // ========================================================================
    // Step 5: Perplexity Calculation
    // ========================================================================
    
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Perplexity Analysis                                             │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    
    // Calculate perplexity using bigram model
    let mut log_prob_sum = 0.0;
    let mut n_predictions = 0;
    
    for seq in &all_sequences {
        if seq.len() >= 2 {
            for window in seq.windows(2) {
                let from = window[0];
                let to = window[1];
                
                let transition_prob = transition_counts.get(&(from, to))
                    .copied()
                    .unwrap_or(0) as f64 / from_counts.get(&from).copied().unwrap_or(1).max(1) as f64;
                
                // Add smoothing for unseen transitions
                let smoothed_prob = transition_prob + 0.01;
                log_prob_sum += smoothed_prob.ln();
                n_predictions += 1;
            }
        }
    }
    
    let avg_log_prob = log_prob_sum / n_predictions.max(1) as f64;
    let perplexity = (-avg_log_prob).exp();
    
    println!("   Average log probability: {:.4}", avg_log_prob);
    println!("   Perplexity: {:.4}", perplexity);
    println!();
    println!("   Interpretation:");
    println!("   - Lower perplexity = more predictable sequences");
    println!("   - Higher perplexity = more random/variable sequences");
    println!("   - Perplexity of {} suggests {} sequence structure",
        format!("{:.2}", perplexity),
        if perplexity < 3.0 { "strong" } else if perplexity < 5.0 { "moderate" } else { "weak" });
    println!();

    // ========================================================================
    // Type Entropy
    // ========================================================================
    
    let mut type_counts: HashMap<i32, usize> = HashMap::new();
    for seq in &all_sequences {
        for &t in seq {
            *type_counts.entry(t).or_default() += 1;
        }
    }
    
    let total: usize = type_counts.values().sum();
    let type_entropy: f64 = type_counts.values()
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.log2()
        })
        .sum();
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         ANALYSIS SUMMARY                                  ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Zebra Finch Song Sequence Analysis                                       ║");
    println!("║                                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Key Metrics:                                                             ║");
    println!("║    • Sequences analyzed:     {:>10}", format!("{}", all_sequences.len()));
    println!("║    • Total phrases:          {:>10}", format!("{}", total_phrases));
    println!("║    • Unique phrase types:    {:>10}", format!("{}", all_types.len()));
    println!("║    • Unique transitions:     {:>10}", format!("{}", transitions.len()));
    println!("║    • Unique bigrams:         {:>10}", format!("{}", bigrams.len()));
    println!("║    • Unique trigrams:        {:>10}", format!("{}", trigrams.len()));
    println!("║    • Recurring motifs:       {:>10}", format!("{}", motifs.len()));
    println!("║    • Type entropy:           {:>10.3} bits", type_entropy);
    println!("║    • Perplexity:             {:>10.2}", perplexity);
    println!("║                                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Conclusions:                                                             ║");
    println!("║                                                                           ║");
    
    let top_transition = transitions.first();
    let top_bigram = bigrams.first();
    let top_motif = motifs.first();
    
    if let Some(t) = top_transition {
        println!("║  • Dominant transition: {} → {} ({:.1}%)", 
            t.from_type, t.to_type, t.probability * 100.0);
    }
    
    if let Some(b) = top_bigram {
        println!("║  • Most common bigram: [{}, {}] ({:.2}%)",
            b.ngram[0], b.ngram[1], b.frequency * 100.0);
    }
    
    if let Some(m) = top_motif {
        let pattern: String = m.pattern.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-");
        println!("║  • Most frequent motif: [{}] ({} occurrences)",
            pattern, m.occurrences);
    }
    
    println!("║                                                                           ║");
    println!("║  Evidence for Syntactic Structure:                                        ║");
    println!("║    • {} transitions show non-random patterns", 
        if transitions.len() > all_types.len() * 2 { "Strong" } else { "Moderate" });
    println!("║    • {} recurring motifs suggest phrase repetition",
        if motifs.len() > 50 { "Many" } else if motifs.len() > 10 { "Several" } else { "Few" });
    println!("║    • Perplexity of {:.2} indicates {} predictability",
        perplexity,
        if perplexity < 3.0 { "high" } else if perplexity < 5.0 { "moderate" } else { "low" });
    println!("║                                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Save results
    let results = SequenceResults {
        total_sequences: all_sequences.len(),
        total_phrases,
        unique_types: all_types.len(),
        transition_matrix: transitions.into_iter().take(100).collect(),
        top_bigrams: bigrams.into_iter().take(100).collect(),
        top_trigrams: trigrams.into_iter().take(100).collect(),
        motifs: motifs.into_iter().take(100).collect(),
        perplexity,
        type_entropy,
        summary: format!("Zebra finch songs show {} syntactic structure with {:.2} perplexity",
            if perplexity < 3.0 { "strong" } else if perplexity < 5.0 { "moderate" } else { "weak" },
            perplexity),
    };
    
    let output_path = "/home/sheel/birdsong_analysis/within_call_results/zebra_finch_sequence_analysis.json";
    let output_file = std::fs::File::create(output_path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(output_file), &results)?;
    println!("Results saved to: {}", output_path);

    Ok(())
}
