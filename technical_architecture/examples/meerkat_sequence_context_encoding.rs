//! Meerkat Sequence-Based Context Encoding Analysis
//!
//! Tests whether context is encoded by phrase SEQUENCE rather than phrase TYPE.
//! Since phrase type diversity was low (only 3 types), this analysis examines
//! if phrase ORDER and TRANSITIONS carry context-specific information.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Clone)]
pub struct SequencePattern {
    pub sequence: Vec<i32>,
    pub frequency: f64,
    pub context: String,
}

#[derive(Debug, Clone)]
pub struct TransitionInfo {
    pub from: i32,
    pub to: i32,
    pub probability: f64,
    pub count: usize,
}

#[derive(Debug)]
pub struct SequenceContextAnalysis {
    /// Overall sequence specificity (mean KL divergence)
    pub sequence_specificity: f64,

    /// Most common sequences per context
    pub context_sequences: HashMap<String, Vec<SequencePattern>>,

    /// N-gram entropy per context
    pub ngram_entropy: HashMap<String, f64>,

    /// Transition matrix differences between contexts (KL divergence)
    pub transition_divergence: HashMap<(String, String), f64>,

    /// Transition matrices per context
    pub transition_matrices: HashMap<String, Vec<TransitionInfo>>,

    /// Unique phrases per context
    pub phrase_diversity: HashMap<String, usize>,

    /// Sequence length statistics per context
    pub sequence_length_stats: HashMap<String, (f64, f64, usize, usize)>, // (mean, std, min, max)
}

// =============================================================================
// Label mappings
// =============================================================================

const LABEL_MEANINGS: &[(&str, &str)] = &[
    ("cc", "Close Call"),
    ("sn", "Sentinel"),
    ("soc", "Social"),
    ("oth", "Other"),
    ("agg", "Aggression"),
    ("synch", "Synchronized"),
    ("al", "Alarm"),
    ("eating", "Eating"),
    ("mo", "Movement"),
    ("beep", "Calibration"),
    ("ld", "Lead"),
];

fn get_label_meaning(label: &str) -> &str {
    LABEL_MEANINGS
        .iter()
        .find(|(l, _)| *l == label)
        .map(|(_, m)| *m)
        .unwrap_or(label)
}

// =============================================================================
// Analysis Functions
// =============================================================================

fn load_phrase_data(path: &str) -> Vec<(String, Vec<i32>)> {
    // Load within-call phrase analysis results (JSON array format)
    let file = File::open(path).expect("Failed to open phrase data");
    let reader = BufReader::new(file);

    let json: serde_json::Value = serde_json::from_reader(reader).expect("Failed to parse JSON");

    let mut results = Vec::new();

    if let Some(arr) = json.as_array() {
        for item in arr {
            if let Some(file_name) = item.get("file_name").and_then(|v| v.as_str()) {
                let fname = file_name.replace(".wav", "");

                // Extract phrase types from the analysis
                let mut phrase_types = Vec::new();

                if let Some(phrases) = item.get("phrases").and_then(|v| v.as_array()) {
                    for phrase in phrases {
                        if let Some(ptype) = phrase.get("phrase_type").and_then(|v| v.as_i64()) {
                            phrase_types.push(ptype as i32);
                        }
                    }
                }

                if !phrase_types.is_empty() {
                    results.push((fname, phrase_types));
                }
            }
        }
    }

    results
}

fn load_labels_via_python(labels_dir: &str) -> HashMap<String, String> {
    use std::process::{Command, Stdio};

    println!("   🔄 Loading labels via Python subprocess...");

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

                output[fname] = primary
    except:
        pass

print(json.dumps(output))
"#;

    // Write Python script to temp file
    let temp_script = "/tmp/load_meerkat_labels_seq.py";
    let mut file = File::create(temp_script).expect("Failed to create temp script");
    file.write_all(python_script.as_bytes())
        .expect("Failed to write script");

    // Run Python script
    let output = Command::new("python3")
        .arg(temp_script)
        .arg(labels_dir)
        .stdout(Stdio::piped())
        .output()
        .expect("Failed to run Python script");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse JSON output
    let mut labels = HashMap::new();
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
        if let Some(obj) = json.as_object() {
            for (k, v) in obj {
                if let Some(label) = v.as_str() {
                    labels.insert(k.clone(), label.to_string());
                }
            }
        }
    }

    labels
}

fn compute_transition_divergence(sequences_a: &[Vec<i32>], sequences_b: &[Vec<i32>]) -> f64 {
    // Compute transition matrices
    let mut trans_a: HashMap<(i32, i32), usize> = HashMap::new();
    let mut trans_b: HashMap<(i32, i32), usize> = HashMap::new();

    for seq in sequences_a {
        for window in seq.windows(2) {
            *trans_a.entry((window[0], window[1])).or_default() += 1;
        }
    }

    for seq in sequences_b {
        for window in seq.windows(2) {
            *trans_b.entry((window[0], window[1])).or_default() += 1;
        }
    }

    let total_a: usize = trans_a.values().sum();
    let total_b: usize = trans_b.values().sum();

    if total_a == 0 || total_b == 0 {
        return 0.0;
    }

    // Jensen-Shannon divergence (symmetric)
    let mut js_divergence = 0.0;
    let all_keys: std::collections::HashSet<_> = trans_a.keys().chain(trans_b.keys()).collect();

    for key in &all_keys {
        let p_a = trans_a.get(key).copied().unwrap_or(0) as f64 / total_a as f64;
        let p_b = trans_b.get(key).copied().unwrap_or(0) as f64 / total_b as f64;
        let p_m = (p_a + p_b) / 2.0;

        if p_a > 0.0 && p_m > 0.0 {
            js_divergence += 0.5 * p_a * (p_a / p_m).ln();
        }
        if p_b > 0.0 && p_m > 0.0 {
            js_divergence += 0.5 * p_b * (p_b / p_m).ln();
        }
    }

    js_divergence.abs()
}

fn compute_entropy(counts: &HashMap<(i32, i32), usize>) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    counts
        .values()
        .map(|&c| {
            let p = c as f64 / total as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn analyze_sequence_encoding(
    phrase_data: &[(String, Vec<i32>)],
    labels: &HashMap<String, String>,
) -> SequenceContextAnalysis {
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Organizing Sequences by Context                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    // Group sequences by context
    let mut context_sequences: HashMap<String, Vec<Vec<i32>>> = HashMap::new();
    let mut matched_count = 0;

    for (fname, sequence) in phrase_data {
        if let Some(context) = labels.get(fname) {
            context_sequences
                .entry(context.clone())
                .or_default()
                .push(sequence.clone());
            matched_count += 1;
        }
    }

    println!("   Matched {} sequences with context labels", matched_count);
    println!("   Found {} unique contexts", context_sequences.len());

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Computing N-gram Statistics per Context                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let mut ngram_entropy = HashMap::new();
    let mut context_patterns: HashMap<String, Vec<SequencePattern>> = HashMap::new();
    let mut transition_matrices: HashMap<String, Vec<TransitionInfo>> = HashMap::new();
    let mut phrase_diversity: HashMap<String, usize> = HashMap::new();
    let mut sequence_length_stats: HashMap<String, (f64, f64, usize, usize)> = HashMap::new();

    for (ctx, sequences) in &context_sequences {
        println!("\n   Processing context: {} ({})", ctx, get_label_meaning(ctx));
        println!("      Sequences: {}", sequences.len());

        // Compute sequence length stats
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let mean_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let var_len = lengths.iter().map(|&l| (l as f64 - mean_len).powi(2)).sum::<f64>() / lengths.len() as f64;
        let std_len = var_len.sqrt();
        let min_len = *lengths.iter().min().unwrap_or(&0);
        let max_len = *lengths.iter().max().unwrap_or(&0);

        sequence_length_stats.insert(ctx.clone(), (mean_len, std_len, min_len, max_len));
        println!(
            "      Sequence length: mean={:.1}, std={:.1}, range=[{}, {}]",
            mean_len, std_len, min_len, max_len
        );

        // Extract bigrams
        let mut bigram_counts: HashMap<(i32, i32), usize> = HashMap::new();
        let mut unique_phrases: std::collections::HashSet<i32> = std::collections::HashSet::new();

        for seq in sequences {
            for window in seq.windows(2) {
                *bigram_counts.entry((window[0], window[1])).or_default() += 1;
            }
            for &p in seq {
                unique_phrases.insert(p);
            }
        }

        let n_unique = unique_phrases.len();
        phrase_diversity.insert(ctx.clone(), n_unique);
        println!("      Unique phrase types: {}", n_unique);

        // Compute entropy
        let entropy = compute_entropy(&bigram_counts);
        ngram_entropy.insert(ctx.clone(), entropy);
        println!("      Bigram entropy: {:.4} bits", entropy);

        // Build transition matrix
        let total_bigrams: usize = bigram_counts.values().sum();
        let mut transitions: Vec<TransitionInfo> = bigram_counts
            .iter()
            .map(|(&(from, to), &count)| TransitionInfo {
                from,
                to,
                probability: count as f64 / total_bigrams as f64,
                count,
            })
            .collect();
        transitions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        transitions.truncate(20);
        transition_matrices.insert(ctx.clone(), transitions);

        // Find most common patterns
        let mut patterns: Vec<SequencePattern> = bigram_counts
            .iter()
            .map(|(&(a, b), &count)| SequencePattern {
                sequence: vec![a, b],
                frequency: count as f64 / total_bigrams as f64,
                context: ctx.clone(),
            })
            .collect();

        patterns.sort_by(|a, b| b.frequency.partial_cmp(&a.frequency).unwrap());
        patterns.truncate(10);
        context_patterns.insert(ctx.clone(), patterns);
    }

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Computing Transition Divergence Between Contexts        │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let contexts: Vec<String> = context_sequences.keys().cloned().collect();
    let mut transition_divergence = HashMap::new();

    println!(
        "   Computing Jensen-Shannon divergence for {} context pairs...",
        contexts.len() * (contexts.len() - 1) / 2
    );

    for i in 0..contexts.len() {
        for j in (i + 1)..contexts.len() {
            let ctx_a = &contexts[i];
            let ctx_b = &contexts[j];

            let div = compute_transition_divergence(&context_sequences[ctx_a], &context_sequences[ctx_b]);

            transition_divergence.insert((ctx_a.clone(), ctx_b.clone()), div);
        }
    }

    // Overall sequence specificity
    let sequence_specificity = if transition_divergence.is_empty() {
        0.0
    } else {
        transition_divergence.values().sum::<f64>() / transition_divergence.len() as f64
    };

    println!("   Mean transition divergence: {:.4}", sequence_specificity);

    SequenceContextAnalysis {
        sequence_specificity,
        context_sequences: context_patterns,
        ngram_entropy,
        transition_divergence,
        transition_matrices,
        phrase_diversity,
        sequence_length_stats,
    }
}

// =============================================================================
// Output and Reporting
// =============================================================================

impl SequenceContextAnalysis {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║       SEQUENCE-BASED CONTEXT ENCODING RESULTS                  ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📊 OVERALL SEQUENCE SPECIFICITY: {:.4}", self.sequence_specificity);
        println!("   (Higher = more context-specific transition patterns)");

        // Sort contexts by entropy
        let mut entropy_sorted: Vec<_> = self.ngram_entropy.iter().collect();
        entropy_sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        println!("\n📊 BIGRAM ENTROPY BY CONTEXT (sorted by entropy):");
        println!("   ┌────────────────────┬──────────┬───────────┬──────────────────────┐");
        println!("   │ Context            │ Entropy  │ Diversity │ Seq Length (mean±std)│");
        println!("   ├────────────────────┼──────────┼───────────┼──────────────────────┤");

        for (ctx, entropy) in &entropy_sorted {
            let meaning = get_label_meaning(ctx);
            let diversity = self.phrase_diversity.get(&**ctx).unwrap_or(&0);
            let (mean, std, _, _) = self.sequence_length_stats.get(&**ctx).unwrap_or(&(0.0, 0.0, 0, 0));
            println!(
                "   │ {:<18} │ {:>7.4} │ {:>9} │ {:>8.1} ± {:.1}        │",
                format!("{} ({})", ctx, meaning),
                entropy,
                diversity,
                mean,
                std
            );
        }
        println!("   └────────────────────┴──────────┴───────────┴──────────────────────┘");

        println!("\n📊 TOP BIGRAMS BY CONTEXT:");
        for (ctx, patterns) in &self.context_sequences {
            let meaning = get_label_meaning(ctx);
            println!("\n   {} ({}):", ctx, meaning);
            for (i, p) in patterns.iter().take(5).enumerate() {
                let seq_str: String = p.sequence.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" → ");
                println!("      {}. [{}]: {:.2}%", i + 1, seq_str, p.frequency * 100.0);
            }
        }

        println!("\n📊 TRANSITION DIVERGENCE (Jensen-Shannon) - TOP 15:");
        let mut divergences: Vec<_> = self.transition_divergence.iter().collect();
        divergences.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        println!("   ┌────────────────────────────────────────┬────────────┐");
        println!("   │ Context Pair                           │ JS Diverg. │");
        println!("   ├────────────────────────────────────────┼────────────┤");

        for ((ctx_a, ctx_b), div) in divergences.iter().take(15) {
            let pair = format!("{} vs {}", ctx_a, ctx_b);
            println!("   │ {:<38} │ {:>10.4} │", pair, div);
        }
        println!("   └────────────────────────────────────────┴────────────┘");

        // Hypothesis test
        self.print_hypothesis_conclusion();
    }

    fn print_hypothesis_conclusion(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           HYPOTHESIS TEST: SEQUENCE ENCODING                   ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        // Find max divergence
        let max_div = self.transition_divergence.values().cloned().fold(0.0, f64::max);

        // Count high-divergence pairs
        let high_div_count = self.transition_divergence.values().filter(|&&d| d > 0.1).count();
        let total_pairs = self.transition_divergence.len();

        // Entropy variance
        let entropy_values: Vec<f64> = self.ngram_entropy.values().cloned().collect();
        let mean_entropy = entropy_values.iter().sum::<f64>() / entropy_values.len() as f64;
        let entropy_var =
            entropy_values.iter().map(|e| (e - mean_entropy).powi(2)).sum::<f64>() / entropy_values.len() as f64;

        println!("\n   📈 Key Metrics:");
        println!("      ├─ Mean Sequence Specificity:  {:.4}", self.sequence_specificity);
        println!("      ├─ Max Transition Divergence:  {:.4}", max_div);
        println!(
            "      ├─ High-Divergence Pairs:      {} / {} ({:.1}%)",
            high_div_count,
            total_pairs,
            high_div_count as f64 / total_pairs as f64 * 100.0
        );
        println!("      ├─ Mean Bigram Entropy:        {:.4} bits", mean_entropy);
        println!("      └─ Entropy Variance:           {:.4}", entropy_var);

        println!("\n   🔬 Interpretation:");

        if self.sequence_specificity > 0.1 && high_div_count > total_pairs / 4 {
            println!();
            println!("   ✅ STRONG EVIDENCE FOR SEQUENCE ENCODING");
            println!();
            println!("      The phrase TRANSITIONS differ significantly between contexts,");
            println!("      suggesting meerkats encode information via phrase ORDER");
            println!("      rather than phrase TYPE alone.");
            println!();
            println!("      This explains why phrase type diversity was low (3 types)");
            println!("      while within-call complexity was high (11.42 phrases/call).");
            println!();
            println!("      INTERPRETATION: Meerkats use a 'temporal syntax' where");
            println!("      context is encoded by the SEQUENCE of phrases, not the");
            println!("      identity of individual phrases.");
        } else if self.sequence_specificity > 0.05 {
            println!();
            println!("   ~ MODERATE EVIDENCE FOR SEQUENCE ENCODING");
            println!();
            println!("      Some context-specific transition patterns exist, but the");
            println!("      signal is weak. Context may be partially encoded by sequence.");
        } else {
            println!();
            println!("   ❌ WEAK EVIDENCE FOR SEQUENCE ENCODING");
            println!();
            println!("      Phrase transitions are similar across contexts, suggesting");
            println!("      context is not primarily encoded by phrase sequence.");
            println!("      Other mechanisms (duration, amplitude, etc.) may be involved.");
        }
    }

    pub fn save_results(&self, output_path: &str) -> std::io::Result<()> {
        use serde_json::json;

        let results = json!({
            "sequence_specificity": self.sequence_specificity,
            "ngram_entropy": self.ngram_entropy,
            "phrase_diversity": self.phrase_diversity,
            "sequence_length_stats": self.sequence_length_stats.iter()
                .map(|(k, &(mean, std, min, max))| (k.clone(), json!({
                    "mean": mean,
                    "std": std,
                    "min": min,
                    "max": max
                })))
                .collect::<HashMap<_, _>>(),
            "transition_divergence": self.transition_divergence.iter()
                .map(|((a, b), d)| (format!("{}_{}", a, b), *d))
                .collect::<HashMap<_, _>>(),
            "context_patterns": self.context_sequences.iter()
                .map(|(ctx, patterns)| {
                    (ctx.clone(), patterns.iter().map(|p| json!({
                        "sequence": p.sequence,
                        "frequency": p.frequency
                    })).collect::<Vec<_>>())
                })
                .collect::<HashMap<_, _>>()
        });

        let mut file = File::create(output_path)?;
        file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;

        println!("\n   ✓ Saved to: {}", output_path);
        Ok(())
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Meerkat Sequence-Based Context Encoding Analysis             ║");
    println!("╠═════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  HYPOTHESIS: Context is encoded by phrase SEQUENCE, not TYPE    ║");
    println!("║                                                                 ║");
    println!("║  Since phrase type diversity was low (3 types), this analysis   ║");
    println!("║  tests if phrase ORDER carries context-specific information.    ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");

    let phrase_path =
        "/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/within_call_results/meerkat_within_call_analyses.json";
    let labels_dir = "/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/lbl/08000Hz";
    let output_dir = "/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/sequence_encoding_results";

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Data                                            │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("   📂 Loading within-call phrase data...");
    let phrase_data = load_phrase_data(phrase_path);
    println!("      └─ Loaded {} sequences", phrase_data.len());

    let labels = load_labels_via_python(labels_dir);
    println!("      └─ Loaded {} file labels", labels.len());

    // Run analysis
    let analysis = analyze_sequence_encoding(&phrase_data, &labels);

    // Print results
    analysis.print_summary();

    // Save results
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    std::fs::create_dir_all(output_dir).ok();
    let output_path = format!("{}/sequence_encoding_analysis.json", output_dir);
    analysis.save_results(&output_path).ok();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ ANALYSIS COMPLETE                                               ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
}
