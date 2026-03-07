// Within-Vocalization Sequence Analysis from Phase 0 Results
// ============================================================
//
// This example analyzes the sequence patterns WITHIN individual vocalizations
// using the existing Phase 0 clustering results (all_segments.json).
//
// Key questions:
// 1. Are phrases ordered consistently within vocalizations?
// 2. Do different contexts use different sequential patterns?
// 3. Are there "syntactic rules" governing phrase order?
//
// Usage: cargo run --release --example bat_within_vocalization_from_p0

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Within-Vocalization Sequence Analysis (from Phase 0 Results)         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let phase0_dir = data_dir.join("phase0_twolevel_hdbscan_results");
    let results_dir = data_dir.join("within_vocalization_results");
    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Phase 0 segments with timing
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Phase 0 Segment Data                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let segments = load_segments(&phase0_dir)?;
    println!(
        "   📂 Loaded {} segments from {} files",
        segments.len(),
        segments.keys().collect::<HashSet<_>>().len()
    );
    println!();

    // ========================================================================
    // Step 2: Load annotations for context mapping
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Loading Annotations                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let annotations = load_annotations(data_dir.join("annotations.csv"))?;
    println!("   📂 Loaded {} file → context mappings", annotations.len());
    println!();

    // ========================================================================
    // Step 3: Analyze Within-Vocalization Sequences
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Analyzing Within-Vocalization Sequences                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let analysis = analyze_within_vocalization_sequences(&segments, &annotations)?;

    // ========================================================================
    // Step 4: Display Results
    // ========================================================================

    display_results(&analysis);

    // ========================================================================
    // Step 5: Save Results
    // ========================================================================

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Saving Results                                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let report_path = results_dir.join("within_vocalization_analysis.json");
    fs::write(&report_path, serde_json::to_string_pretty(&analysis)?)?;
    println!("   💾 Results saved: {}", report_path.display());
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, serde::Serialize)]
struct Segment {
    segment_id: usize,
    file_name: String,
    start_time_ms: f64,
    end_time_ms: f64,
    duration_ms: f64,
    level1_cluster_id: i32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct FileSequence {
    file_name: String,
    context: Option<i32>,
    segments: Vec<Segment>,
    cluster_sequence: Vec<i32>,
    sequence_length: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Transition {
    from_cluster: i32,
    to_cluster: i32,
    count: usize,
    contexts: HashSet<i32>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct SequencePattern {
    pattern: Vec<i32>,
    occurrences: usize,
    files: Vec<String>,
    contexts: HashSet<i32>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct WithinVocalizationAnalysis {
    file_sequences: Vec<FileSequence>,
    transitions: Vec<Transition>,
    common_patterns: Vec<SequencePattern>,
    context_patterns: HashMap<i32, ContextSequenceStats>,
    cluster_statistics: ClusterStatistics,
    entropy_analysis: EntropyAnalysis,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ContextSequenceStats {
    context_id: i32,
    num_files: usize,
    avg_sequence_length: f64,
    sequence_length_std: f64,
    unique_bigrams: usize,
    total_bigrams: usize,
    bigram_entropy: f64,
    top_bigrams: Vec<(Vec<i32>, usize)>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClusterStatistics {
    unique_clusters: usize,
    cluster_frequencies: HashMap<i32, usize>,
    most_common_starting_clusters: Vec<(i32, usize)>,
    most_common_ending_clusters: Vec<(i32, usize)>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct EntropyAnalysis {
    overall_bigram_entropy: f64,
    overall_trigram_entropy: f64,
    context_bigram_entropies: HashMap<i32, f64>,
    predictability_index: f64,
}

// ============================================================================
// Data Loading
// ============================================================================

#[derive(Debug, Clone)]
struct Annotation {
    context: i32,
    file_name: String,
}

fn load_annotations(path: impl AsRef<Path>) -> Result<HashMap<String, i32>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut mapping = HashMap::new();

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            if let Ok(context) = parts[2].parse::<i32>() {
                let file_name = parts[7].trim().to_string();
                mapping.insert(file_name, context);
            }
        }
    }

    Ok(mapping)
}

fn load_segments(phase0_dir: &Path) -> Result<HashMap<String, Vec<Segment>>, Box<dyn std::error::Error>> {
    let segments_path = phase0_dir.join("all_segments.json");
    let segments_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&segments_path)?)?;

    let mut file_segments: HashMap<String, Vec<Segment>> = HashMap::new();

    if let Some(arr) = segments_json.as_array() {
        for segment in arr {
            if let (Some(segment_id), Some(file_name), Some(start), Some(end), Some(cluster_id)) = (
                segment["segment_id"].as_u64(),
                segment["file_name"].as_str(),
                segment["start_time_ms"].as_f64(),
                segment["end_time_ms"].as_f64(),
                segment["level1_cluster_id"].as_i64(),
            ) {
                let duration = end - start;

                // Skip noise segments
                if cluster_id == -1 {
                    continue;
                }

                let seg = Segment {
                    segment_id: segment_id as usize,
                    file_name: file_name.to_string(),
                    start_time_ms: start,
                    end_time_ms: end,
                    duration_ms: duration,
                    level1_cluster_id: cluster_id as i32,
                };

                file_segments
                    .entry(file_name.to_string())
                    .or_insert_with(Vec::new)
                    .push(seg);
            }
        }
    }

    // Sort segments by start time within each file
    for segments in file_segments.values_mut() {
        segments.sort_by(|a, b| a.start_time_ms.partial_cmp(&b.start_time_ms).unwrap());
    }

    Ok(file_segments)
}

// ============================================================================
// Analysis Functions
// ============================================================================

fn analyze_within_vocalization_sequences(
    file_segments: &HashMap<String, Vec<Segment>>,
    annotations: &HashMap<String, i32>,
) -> Result<WithinVocalizationAnalysis, Box<dyn std::error::Error>> {
    println!("   📊 Processing {} files...", file_segments.len());

    // Build file sequences
    let mut file_sequences: Vec<FileSequence> = Vec::new();
    let mut context_sequences: HashMap<i32, Vec<Vec<i32>>> = HashMap::new();
    let mut all_bigrams: Vec<Vec<i32>> = Vec::new();
    let mut all_trigrams: Vec<Vec<i32>> = Vec::new();
    let mut context_bigrams: HashMap<i32, Vec<Vec<i32>>> = HashMap::new();

    for (file_name, segments) in file_segments {
        let context = annotations.get(file_name).copied();

        let cluster_sequence: Vec<i32> = segments.iter().map(|s| s.level1_cluster_id).collect();

        if let Some(ctx) = context {
            context_sequences
                .entry(ctx)
                .or_insert_with(Vec::new)
                .push(cluster_sequence.clone());

            // Extract n-grams for this context
            for window in cluster_sequence.windows(2) {
                let bigram = window.to_vec();
                all_bigrams.push(bigram.clone());
                context_bigrams.entry(ctx).or_insert_with(Vec::new).push(bigram);
            }
            for window in cluster_sequence.windows(3) {
                all_trigrams.push(window.to_vec());
            }
        }

        file_sequences.push(FileSequence {
            file_name: file_name.clone(),
            context,
            segments: segments.clone(),
            cluster_sequence: cluster_sequence.clone(),
            sequence_length: cluster_sequence.len(),
        });
    }

    println!("   ✅ Processed {} file sequences", file_sequences.len());
    println!();

    // Analyze transitions
    let transitions = analyze_transitions(&file_sequences)?;

    // Find common patterns
    let common_patterns = find_common_patterns(&file_sequences, 3)?;

    // Analyze context-specific patterns
    let context_patterns = analyze_context_patterns(&context_sequences)?;

    // Cluster statistics
    let cluster_statistics = analyze_cluster_statistics(&file_sequences);

    // Entropy analysis
    let entropy_analysis = analyze_entropy(&all_bigrams, &all_trigrams, &context_bigrams)?;

    println!("   📊 Analysis Summary:");
    println!("      Unique transitions: {}", transitions.len());
    println!("      Common patterns found: {}", common_patterns.len());
    println!("      Contexts analyzed: {}", context_patterns.len());
    println!();

    Ok(WithinVocalizationAnalysis {
        file_sequences,
        transitions,
        common_patterns,
        context_patterns,
        cluster_statistics,
        entropy_analysis,
    })
}

fn analyze_transitions(file_sequences: &[FileSequence]) -> Result<Vec<Transition>, Box<dyn std::error::Error>> {
    let mut transition_map: HashMap<(i32, i32), Transition> = HashMap::new();

    for seq in file_sequences {
        for window in seq.cluster_sequence.windows(2) {
            let from = window[0];
            let to = window[1];

            let entry = transition_map.entry((from, to)).or_insert_with(|| Transition {
                from_cluster: from,
                to_cluster: to,
                count: 0,
                contexts: HashSet::new(),
            });

            entry.count += 1;
            if let Some(ctx) = seq.context {
                entry.contexts.insert(ctx);
            }
        }
    }

    let mut transitions: Vec<Transition> = transition_map.into_values().collect();
    transitions.sort_by(|a, b| b.count.cmp(&a.count));

    println!("   📊 Found {} unique transitions", transitions.len());

    Ok(transitions)
}

fn find_common_patterns(
    file_sequences: &[FileSequence],
    min_occurrences: usize,
) -> Result<Vec<SequencePattern>, Box<dyn std::error::Error>> {
    let mut pattern_counts: HashMap<Vec<i32>, SequencePattern> = HashMap::new();

    // Look for patterns of length 2-4
    for seq in file_sequences {
        for pattern_len in 2..=4 {
            for window in seq.cluster_sequence.windows(pattern_len) {
                let pattern = window.to_vec();

                let entry = pattern_counts
                    .entry(pattern.clone())
                    .or_insert_with(|| SequencePattern {
                        pattern: pattern.clone(),
                        occurrences: 0,
                        files: Vec::new(),
                        contexts: HashSet::new(),
                    });

                entry.occurrences += 1;
                entry.files.push(seq.file_name.clone());
                if let Some(ctx) = seq.context {
                    entry.contexts.insert(ctx);
                }
            }
        }
    }

    let mut patterns: Vec<SequencePattern> = pattern_counts
        .into_values()
        .filter(|p| p.occurrences >= min_occurrences && p.contexts.len() >= 2)
        .collect();

    // Sort by occurrences and number of contexts
    patterns.sort_by(|a, b| {
        b.occurrences
            .cmp(&a.occurrences)
            .then_with(|| b.contexts.len().cmp(&a.contexts.len()))
    });

    println!(
        "   📊 Found {} patterns with {}+ occurrences in 2+ contexts",
        patterns.len(),
        min_occurrences
    );

    Ok(patterns)
}

fn analyze_context_patterns(
    context_sequences: &HashMap<i32, Vec<Vec<i32>>>,
) -> Result<HashMap<i32, ContextSequenceStats>, Box<dyn std::error::Error>> {
    let mut context_stats: HashMap<i32, ContextSequenceStats> = HashMap::new();

    for (&context_id, sequences) in context_sequences {
        let num_files = sequences.len();

        // Calculate average sequence length
        let lengths: Vec<f64> = sequences.iter().map(|s| s.len() as f64).collect();
        let avg_length = lengths.iter().sum::<f64>() / lengths.len() as f64;

        // Calculate standard deviation
        let variance = lengths.iter().map(|&l| (l - avg_length).powi(2)).sum::<f64>() / lengths.len() as f64;
        let std_dev = variance.sqrt();

        // Analyze bigrams
        let mut bigram_counts: HashMap<Vec<i32>, usize> = HashMap::new();
        let mut total_bigrams = 0;

        for seq in sequences {
            for window in seq.windows(2) {
                *bigram_counts.entry(window.to_vec()).or_insert(0) += 1;
                total_bigrams += 1;
            }
        }

        let unique_bigrams = bigram_counts.len();

        // Calculate bigram entropy
        let bigram_entropy = if total_bigrams > 0 {
            let mut entropy = 0.0;
            for &count in bigram_counts.values() {
                let p = count as f64 / total_bigrams as f64;
                if p > 0.0 {
                    entropy -= p * p.log2();
                }
            }
            entropy
        } else {
            0.0
        };

        // Get top bigrams
        let mut top_bigrams: Vec<(Vec<i32>, usize)> = bigram_counts.into_iter().collect();
        top_bigrams.sort_by(|a, b| b.1.cmp(&a.1));
        top_bigrams.truncate(10);

        context_stats.insert(
            context_id,
            ContextSequenceStats {
                context_id,
                num_files,
                avg_sequence_length: avg_length,
                sequence_length_std: std_dev,
                unique_bigrams,
                total_bigrams,
                bigram_entropy,
                top_bigrams,
            },
        );
    }

    println!("   📊 Analyzed {} contexts", context_stats.len());

    Ok(context_stats)
}

fn analyze_cluster_statistics(file_sequences: &[FileSequence]) -> ClusterStatistics {
    let mut cluster_frequencies: HashMap<i32, usize> = HashMap::new();
    let mut starting_clusters: HashMap<i32, usize> = HashMap::new();
    let mut ending_clusters: HashMap<i32, usize> = HashMap::new();

    for seq in file_sequences {
        if let Some(&first) = seq.cluster_sequence.first() {
            *starting_clusters.entry(first).or_insert(0) += 1;
            *cluster_frequencies.entry(first).or_insert(0) += 1;
        }

        for &cluster in &seq.cluster_sequence {
            *cluster_frequencies.entry(cluster).or_insert(0) += 1;
        }

        if let Some(&last) = seq.cluster_sequence.last() {
            *ending_clusters.entry(last).or_insert(0) += 1;
        }
    }

    let mut most_common_starting: Vec<(i32, usize)> = starting_clusters.into_iter().collect();
    most_common_starting.sort_by(|a, b| b.1.cmp(&a.1));
    most_common_starting.truncate(10);

    let mut most_common_ending: Vec<(i32, usize)> = ending_clusters.into_iter().collect();
    most_common_ending.sort_by(|a, b| b.1.cmp(&a.1));
    most_common_ending.truncate(10);

    let unique_clusters = cluster_frequencies.len();

    ClusterStatistics {
        unique_clusters,
        cluster_frequencies,
        most_common_starting_clusters: most_common_starting,
        most_common_ending_clusters: most_common_ending,
    }
}

fn analyze_entropy(
    all_bigrams: &[Vec<i32>],
    all_trigrams: &[Vec<i32>],
    context_bigrams: &HashMap<i32, Vec<Vec<i32>>>,
) -> Result<EntropyAnalysis, Box<dyn std::error::Error>> {
    // Overall bigram entropy
    let overall_bigram_entropy = calculate_entropy(all_bigrams);

    // Overall trigram entropy
    let overall_trigram_entropy = calculate_entropy(all_trigrams);

    // Per-context bigram entropy
    let mut context_bigram_entropies: HashMap<i32, f64> = HashMap::new();
    for (&context, bigrams) in context_bigrams {
        context_bigram_entropies.insert(context, calculate_entropy(bigrams));
    }

    // Predictability index: how much does context reduce uncertainty?
    let avg_context_entropy = if context_bigram_entropies.is_empty() {
        0.0
    } else {
        context_bigram_entropies.values().sum::<f64>() / context_bigram_entropies.len() as f64
    };

    let predictability_index = if overall_bigram_entropy > 0.0 {
        (overall_bigram_entropy - avg_context_entropy) / overall_bigram_entropy
    } else {
        0.0
    };

    Ok(EntropyAnalysis {
        overall_bigram_entropy,
        overall_trigram_entropy,
        context_bigram_entropies,
        predictability_index,
    })
}

fn calculate_entropy<T: std::hash::Hash + Eq>(items: &[Vec<T>]) -> f64 {
    if items.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<&[T], usize> = HashMap::new();
    for item in items {
        *counts.entry(item.as_slice()).or_insert(0) += 1;
    }

    let total = items.len() as f64;
    let mut entropy = 0.0;

    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

// ============================================================================
// Display Functions
// ============================================================================

fn display_results(analysis: &WithinVocalizationAnalysis) {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    WITHIN-VOCALIZATION ANALYSIS RESULTS                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Cluster statistics
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Cluster Statistics                                                       │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   Unique clusters: {}", analysis.cluster_statistics.unique_clusters);
    println!("   Total files analyzed: {}", analysis.file_sequences.len());
    println!();

    println!("   Most Common Starting Clusters:");
    for (i, (cluster, count)) in analysis
        .cluster_statistics
        .most_common_starting_clusters
        .iter()
        .enumerate()
        .take(10)
    {
        println!(
            "      {:2}. Cluster {:>3}: starts {} sequences ({:.1}%)",
            i + 1,
            cluster,
            count,
            *count as f64 * 100.0 / analysis.file_sequences.len() as f64
        );
    }
    println!();

    println!("   Most Common Ending Clusters:");
    for (i, (cluster, count)) in analysis
        .cluster_statistics
        .most_common_ending_clusters
        .iter()
        .enumerate()
        .take(10)
    {
        println!(
            "      {:2}. Cluster {:>3}: ends {} sequences ({:.1}%)",
            i + 1,
            cluster,
            count,
            *count as f64 * 100.0 / analysis.file_sequences.len() as f64
        );
    }
    println!();

    // Top transitions
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Top 20 Most Common Transitions                                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "{:<5} {:<15} {:<15} {:>12} {:>20}",
        "Rank", "From", "To", "Count", "Contexts"
    );
    println!("{}", "-".repeat(75));

    for (i, trans) in analysis.transitions.iter().take(20).enumerate() {
        let ctx_str = if trans.contexts.is_empty() {
            "N/A".to_string()
        } else {
            format!("{} contexts", trans.contexts.len())
        };

        println!(
            "{:<5} {:<15} {:<15} {:>12} {:>20}",
            i + 1,
            format!("Cluster {}", trans.from_cluster),
            format!("Cluster {}", trans.to_cluster),
            trans.count,
            ctx_str
        );
    }
    println!();

    // Common patterns
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Top 20 Cross-Context Sequential Patterns                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "{:<5} {:<30} {:>12} {:>12} {:>15}",
        "Rank", "Pattern", "Occurrences", "Files", "Contexts"
    );
    println!("{}", "-".repeat(80));

    for (i, pattern) in analysis.common_patterns.iter().take(20).enumerate() {
        let pattern_str = format!("{:?}", pattern.pattern);
        println!(
            "{:<5} {:<30} {:>12} {:>12} {:>15}",
            i + 1,
            truncate_string(&pattern_str, 28),
            pattern.occurrences,
            pattern.files.len(),
            format!("{} ctx", pattern.contexts.len())
        );
    }
    println!();

    // Context-specific patterns
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Per-Context Sequence Statistics                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut context_entries: Vec<_> = analysis.context_patterns.values().collect();
    context_entries.sort_by(|a, b| b.num_files.cmp(&a.num_files));

    println!(
        "{:<10} {:>12} {:>15} {:>15} {:>12} {:>12} {:>15}",
        "Context", "Files", "Avg Length", "Std Dev", "Bigrams", "Unique", "Entropy"
    );
    println!("{}", "-".repeat(100));

    for stats in context_entries {
        println!(
            "{:<10} {:>12} {:>15.2} {:>15.2} {:>12} {:>12} {:>15.3}",
            stats.context_id,
            stats.num_files,
            stats.avg_sequence_length,
            stats.sequence_length_std,
            stats.total_bigrams,
            stats.unique_bigrams,
            stats.bigram_entropy
        );
    }
    println!();

    // Entropy analysis
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Entropy Analysis (Predictability)                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "   Overall bigram entropy: {:.3} bits",
        analysis.entropy_analysis.overall_bigram_entropy
    );
    println!(
        "   Overall trigram entropy: {:.3} bits",
        analysis.entropy_analysis.overall_trigram_entropy
    );
    println!();

    println!("   Context-specific entropies:");
    let mut ctx_entropies: Vec<_> = analysis.entropy_analysis.context_bigram_entropies.iter().collect();
    ctx_entropies.sort_by(|a, b| a.0.cmp(b.0));

    for (ctx, entropy) in ctx_entropies {
        println!("      Context {:>3}: {:.3} bits", ctx, entropy);
    }
    println!();

    println!(
        "   Predictability Index: {:.3}",
        analysis.entropy_analysis.predictability_index
    );
    println!("      (>0 = context reduces uncertainty, <0 = context adds uncertainty)");
    println!();

    // Interpretation
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Scientific Interpretation                                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let avg_len = analysis
        .context_patterns
        .values()
        .map(|s| s.avg_sequence_length)
        .sum::<f64>()
        / analysis.context_patterns.len() as f64;

    let avg_entropy = analysis.entropy_analysis.context_bigram_entropies.values().sum::<f64>()
        / analysis.entropy_analysis.context_bigram_entropies.len() as f64;

    println!("   Average sequence length: {:.2} phrases per vocalization", avg_len);
    println!("   Average bigram entropy: {:.3} bits", avg_entropy);
    println!();

    if avg_entropy < 3.0 {
        println!("   ✅ LOW ENTROPY: Phrase sequences are PREDICTABLE");
        println!("      → Strong evidence for SYNTACTIC RULES");
        println!("      → Specific phrase transitions are preferred");
    } else if avg_entropy < 6.0 {
        println!("   ⚠️  MEDIUM ENTROPY: Phrase sequences are MODERATELY PREDICTABLE");
        println!("      → Suggests FLEXIBLE SYNTAX");
        println!("      → Some patterns but many possibilities");
    } else {
        println!("   ❌ HIGH ENTROPY: Phrase sequences are UNPREDICTABLE");
        println!("      → NO evidence for syntactic rules");
        println!("      → Phrase transitions appear random");
    }
    println!();

    if analysis.entropy_analysis.predictability_index > 0.1 {
        println!(
            "   ✅ POSITIVE predictability index ({:.3}):",
            analysis.entropy_analysis.predictability_index
        );
        println!("      → Knowing the context REDUCES uncertainty about next phrase");
        println!("      → Supports context-specific syntax");
    } else if analysis.entropy_analysis.predictability_index < -0.1 {
        println!(
            "   ❌ NEGATIVE predictability index ({:.3}):",
            analysis.entropy_analysis.predictability_index
        );
        println!("      → Context knowledge does NOT help prediction");
        println!("      → Syntax is NOT context-specific");
    } else {
        println!(
            "   ⚠️  NEUTRAL predictability index ({:.3}):",
            analysis.entropy_analysis.predictability_index
        );
        println!("      → Context has minimal effect on sequence predictability");
    }
    println!();

    if analysis.common_patterns.len() > 10 {
        println!("   ✅ {} cross-context patterns found:", analysis.common_patterns.len());
        println!("      → Some sequences are shared across behavioral contexts");
        println!("      → May represent UNIVERSAL syntactic structures");
    } else {
        println!(
            "   ⚠️  Only {} cross-context patterns found:",
            analysis.common_patterns.len()
        );
        println!("      → Few universal sequences across contexts");
        println!("      → Each context may have its own patterns");
    }
    println!();
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
