// Phase 2: Advanced Sequence Analysis - Testing Combinatorial Syntax
//
// This example implements five computational methods to test for sentence structures
// and reusable phrases in Egyptian Fruit Bat vocalizations:
//
// 1. Multiple Sequence Alignment (MSA) - Find conserved regions across contexts
// 2. Hidden Markov Models (HMM) - Discover hidden phrase states
// 3. N-Gram Perplexity - Cross-context prediction testing
// 4. Network Motif Analysis - Find recurring structural patterns
// 5. Supervised ML - Test if syntax carries more information than content
//
// Usage: cargo run --release --example phase2_advanced_sequence_analysis

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Phase 2: Advanced Sequence Analysis - Egyptian Fruit Bat                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Testing for Combinatorial Syntax using 5 Computational Methods:           ║");
    println!("║    1. Multiple Sequence Alignment (MSA)                                   ║");
    println!("║    2. Hidden Markov Models (HMM)                                         ║");
    println!("║    3. N-Gram Perplexity (Cross-Context Prediction)                        ║");
    println!("║    4. Network Motif Analysis                                             ║");
    println!("║    5. Supervised Machine Learning                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let phase0_results_dir = data_dir.join("phase0_twolevel_hdbscan_results");
    let results_dir = data_dir.join("phase2_sequence_analysis_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Phase 0 Data (Symbolic Stream)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Phase 0 Symbolic Stream Data                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let file_segments = load_file_segments(&phase0_results_dir)?;
    let total_segments: usize = file_segments.values().map(|v| v.len()).sum();
    println!("   📂 Loaded {} files with {} total segments", file_segments.len(), total_segments);
    println!();

    // ========================================================================
    // Step 2: Load Annotations and Group by Context
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Loading Annotations and Grouping by Context                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let annotations = load_annotations(data_dir.join("annotations.csv"))?;
    println!("   📂 Loaded {} annotations", annotations.len());
    println!();

    // Build file_name -> context mapping
    let mut file_context_map: HashMap<String, i32> = HashMap::new();
    for ann in &annotations {
        let file_key = ann.file_name.trim().to_string();
        file_context_map.insert(file_key, ann.context);
    }

    // Validate: Check for files with annotations but no segments
    let segment_files: HashSet<_> = file_segments.keys().collect();
    let annotation_files: HashSet<_> = file_context_map.keys().collect();
    let missing_files: Vec<_> = annotation_files.difference(&segment_files).collect();
    let matched_files: Vec<_> = annotation_files.intersection(&segment_files).collect();

    println!("   📊 File Matching Statistics:");
    println!("      Files with segments: {}", file_segments.len());
    println!("      Files with annotations: {}", file_context_map.len());
    println!("      Matched files: {}", matched_files.len());
    println!("      Missing files (annotation but no segment): {}", missing_files.len());

    if !missing_files.is_empty() && missing_files.len() <= 20 {
        println!("      Missing files: {:?}", missing_files);
    } else if missing_files.len() > 20 {
        println!("      First 20 missing files: {:?}", missing_files.iter().take(20).collect::<Vec<_>>());
    }
    println!();

    // Group phrases by context (create sequences - one per file)
    let sequences_by_context = group_sequences_by_context(&file_segments, &file_context_map)?;

    println!("   📊 Context Distribution:");
    let mut contexts: Vec<_> = sequences_by_context.iter().collect();
    contexts.sort_by(|a, b| (b.1.len()).cmp(&(a.1.len())));

    for (_i, (ctx, seqs)) in contexts.iter().enumerate().take(10) {
        let total_phrases: usize = seqs.iter().map(|s| s.len()).sum();
        let avg_phrase_count = if seqs.is_empty() { 0.0 } else { total_phrases as f64 / seqs.len() as f64 };
        println!("      Context {}: {} file-sequences, {} total phrases (avg {:.1} per file)",
                 ctx, seqs.len(), total_phrases, avg_phrase_count);
    }
    println!();

    // ========================================================================
    // Step 3: Run Advanced Sequence Analysis Suite
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Running Advanced Sequence Analysis Suite                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    use technical_architecture::sequence_analysis::SequenceAnalysisSuite;

    let suite = SequenceAnalysisSuite::new(data_dir);

    let report = suite.run_full_analysis(&sequences_by_context)?;

    // ========================================================================
    // Step 4: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Saving Analysis Results                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save full report
    let report_path = results_dir.join("sequence_analysis_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("   💾 Report saved: {}", report_path.display());

    // Save sequences by context
    let seq_path = results_dir.join("sequences_by_context.json");
    fs::write(&seq_path, serde_json::to_string_pretty(&sequences_by_context)?)?;
    println!("   💾 Sequences saved: {}", seq_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 SUMMARY OF FINDINGS:                                                  ║");
    println!("║     • MSA conserved regions: {}                                          ║", report.msa_conserved_regions);
    println!("║     • HMM hidden states: {}                                              ║", report.hmm_states);
    println!("║     • Network motifs: {}                                                 ║", report.network_motifs);
    println!("║     • Multi-context motifs: {}                                           ║", report.multi_context_motifs);
    println!("║                                                                           ║");
    if report.ml_improvement > 0.0 {
        println!("║     ✅ Syntax features improve prediction by {:.1}%                    ║", report.ml_improvement * 100.0);
        println!("║        This SUPPORTS the combinatorial syntax hypothesis                ║");
    } else {
        println!("║     ⚠️  Syntax features do not improve prediction                        ║");
        println!("║        This DOES NOT support the combinatorial syntax hypothesis        ║");
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

fn load_annotations(path: impl AsRef<Path>) -> Result<Vec<Annotation>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut annotations = Vec::new();

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

/// Load file-to-segments mapping from all_segments.json
/// Returns: HashMap<file_name, Vec<level1_cluster_id>>
fn load_file_segments(
    results_dir: &Path,
) -> Result<HashMap<String, Vec<i32>>, Box<dyn std::error::Error>> {
    let segments_path = results_dir.join("all_segments.json");
    let segments_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&segments_path)?)?;

    println!("   📖 Parsing segment data...");

    let mut file_segments: HashMap<String, Vec<i32>> = HashMap::new();
    let mut noise_count = 0;
    let mut total_count = 0;

    if let Some(arr) = segments_json.as_array() {
        for segment in arr {
            if let Some(file_name) = segment["file_name"].as_str() {
                let cluster_id = segment["level1_cluster_id"]
                    .as_i64()
                    .unwrap_or(-1) as i32;

                total_count += 1;

                // Skip noise segments for now (can be configured)
                if cluster_id == -1 {
                    noise_count += 1;
                    continue;
                }

                file_segments
                    .entry(file_name.to_string())
                    .or_insert_with(Vec::new)
                    .push(cluster_id);
            }
        }
    }

    println!("   📊 Segment Statistics:");
    println!("      Total segments: {}", total_count);
    println!("      Noise segments (skipped): {}", noise_count);
    println!("      Valid segments: {}", total_count - noise_count);
    println!("      Unique files with valid segments: {}", file_segments.len());

    // Calculate average segments per file
    let avg_segments = if file_segments.is_empty() {
        0.0
    } else {
        total_count as f64 / file_segments.len() as f64
    };
    println!("      Average segments per file: {:.2}", avg_segments);
    println!();

    Ok(file_segments)
}

fn group_sequences_by_context(
    file_segments: &HashMap<String, Vec<i32>>,
    file_context_map: &HashMap<String, i32>,
) -> Result<HashMap<String, Vec<Vec<i32>>>, Box<dyn std::error::Error>> {
    println!("   🔗 Grouping sequences by context (one sequence per file)...");
    println!();

    let mut sequences_by_context: HashMap<String, Vec<Vec<i32>>> = HashMap::new();
    let mut total_files_processed = 0;
    let mut total_phrases_processed = 0;
    let mut skipped_no_context = 0;

    // Process each file independently to create one sequence per file
    for (file_name, cluster_ids) in file_segments {
        // Get the context for this file
        if let Some(&context) = file_context_map.get(file_name) {
            let context_key = context.to_string();

            // Offset cluster IDs to avoid collision with gap marker (-999)
            // and create a sequence from this file's cluster IDs
            let sequence: Vec<i32> = cluster_ids.iter()
                .map(|&id| id + 1000) // Use larger offset (1000) to be safe
                .collect();

            sequences_by_context
                .entry(context_key)
                .or_insert_with(Vec::new)
                .push(sequence);

            total_files_processed += 1;
            total_phrases_processed += cluster_ids.len();
        } else {
            skipped_no_context += 1;
        }
    }

    println!("   📊 Grouping Statistics:");
    println!("      Files processed: {}", total_files_processed);
    println!("      Phrases processed: {}", total_phrases_processed);
    println!("      Files skipped (no context): {}", skipped_no_context);
    println!("      Contexts with data: {}", sequences_by_context.len());
    println!();

    // Validate unique cluster IDs
    let all_ids: HashSet<i32> = file_segments.values()
        .flat_map(|ids| ids.iter().copied())
        .collect();
    println!("   📊 Cluster ID Statistics:");
    println!("      Unique cluster IDs (raw): {}", all_ids.len());
    println!("      Cluster ID range: {} to {}",
             all_ids.iter().min().unwrap_or(&-1),
             all_ids.iter().max().unwrap_or(&-1));
    println!("      After offset (1000): {} to {}",
             all_ids.iter().min().unwrap_or(&-1) + 1000,
             all_ids.iter().max().unwrap_or(&-1) + 1000);
    println!();

    Ok(sequences_by_context)
}
