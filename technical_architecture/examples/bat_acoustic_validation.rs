// Acoustic Cluster Validation & True Phrase Transition Analysis
// ============================================================
//
// This example performs 4 critical analyses:
//
// 1. Examines Level 2 vocabulary structure
// 2. Validates cluster acoustic coherence (within vs between cluster distances)
// 3. Analyzes representative features to detect position bias
// 4. Performs position-independent cluster analysis
//
// Usage: cargo run --release --example bat_acoustic_validation

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Acoustic Cluster Validation & Transition Analysis                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let phase0_dir = data_dir.join("phase0_twolevel_hdbscan_results");
    let results_dir = data_dir.join("acoustic_validation_results");
    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load All Data
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Phase 0 Data                                           │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let segments = load_segments(&phase0_dir)?;
    println!("   📂 Loaded {} segments", segments.len());

    let vocabulary = load_vocabulary(&phase0_dir)?;
    println!("   📂 Loaded {} vocabulary entries", vocabulary.len());

    let annotations = load_annotations(data_dir.join("annotations.csv"))?;
    println!("   📂 Loaded {} annotations", annotations.len());
    println!();

    // ========================================================================
    // ANALYSIS 1: Level 2 Vocabulary Structure
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ ANALYSIS 1: Level 2 Vocabulary Structure                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    analyze_vocabulary_structure(&vocabulary, &segments)?;

    // ========================================================================
    // ANALYSIS 2: Position Bias Detection
    // ========================================================================

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ ANALYSIS 2: Position Bias Detection                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    detect_position_bias(&segments)?;

    // ========================================================================
    // ANALYSIS 3: Acoustic Coherence Validation
    // ========================================================================

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ ANALYSIS 3: Acoustic Coherence Validation                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let acoustic_analysis = validate_acoustic_coherence(&segments)?;

    // ========================================================================
    // ANALYSIS 4: True Phrase Transition Analysis (Position-Independent)
    // ========================================================================

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ ANALYSIS 4: True Phrase Transitions (Position-Independent)              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let transition_analysis =
        analyze_true_transitions(&segments, &annotations, &acoustic_analysis)?;

    // ========================================================================
    // Save Results
    // ========================================================================

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                                          │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let full_results = serde_json::json!({
        "vocabulary_structure": vocabulary,
        "position_bias": "ANALYZED",
        "acoustic_coherence": acoustic_analysis,
        "transitions": transition_analysis
    });

    let results_path = results_dir.join("acoustic_validation_report.json");
    fs::write(&results_path, serde_json::to_string_pretty(&full_results)?)?;
    println!("   💾 Results saved: {}", results_path.display());
    println!();

    // ========================================================================
    // Final Summary & Recommendations
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    FINAL RECOMMENDATIONS                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    print_recommendations(&acoustic_analysis, &transition_analysis);

    Ok(())
}

// ============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Clone)]
struct Segment {
    segment_id: usize,
    file_name: String,
    start_time_ms: f64,
    end_time_ms: f64,
    duration_ms: f64,
    level1_cluster_id: i32,
    features: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct VocabularyEntry {
    vocabulary_id: i32,
    level2_cluster_id: i32,
    phrase_count: usize,
    avg_duration_ms: f64,
    std_duration_ms: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AcousticAnalysis {
    feature_dimension: usize,
    total_segments_analyzed: usize,
    cluster_count: usize,
    within_cluster_distances: Vec<ClusterDistanceStats>,
    between_cluster_distances: Vec<BetweenClusterDistance>,
    silhouette_score: f64,
    coherence_assessment: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClusterDistanceStats {
    cluster_id: i32,
    segment_count: usize,
    mean_distance: f64,
    std_distance: f64,
    min_distance: f64,
    max_distance: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BetweenClusterDistance {
    cluster_a: i32,
    cluster_b: i32,
    mean_distance: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TransitionAnalysis {
    context_specific_patterns: HashMap<i32, ContextPattern>,
    cross_context_similarity: f64,
    entropy_by_context: HashMap<i32, f64>,
    syntax_evidence_score: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ContextPattern {
    context_id: i32,
    num_files: usize,
    dominant_patterns: Vec<DominantPattern>,
    pattern_diversity: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DominantPattern {
    cluster_sequence: Vec<i32>,
    frequency: usize,
    proportion: f64,
}

// ============================================================================
// Data Loading
// ============================================================================

fn load_annotations(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, i32>, Box<dyn std::error::Error>> {
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

fn load_segments(phase0_dir: &Path) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let segments_path = phase0_dir.join("all_segments.json");
    let segments_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&segments_path)?)?;

    let mut segments = Vec::new();

    if let Some(arr) = segments_json.as_array() {
        for segment in arr {
            if let (
                Some(segment_id),
                Some(file_name),
                Some(start),
                Some(end),
                Some(cluster_id),
                Some(features),
            ) = (
                segment["segment_id"].as_u64(),
                segment["file_name"].as_str(),
                segment["start_time_ms"].as_f64(),
                segment["end_time_ms"].as_f64(),
                segment["level1_cluster_id"].as_i64(),
                segment["representative_features"].as_array(),
            ) {
                let duration = end - start;

                // Convert features to Vec<f32>
                let feature_vec: Vec<f32> = features
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();

                segments.push(Segment {
                    segment_id: segment_id as usize,
                    file_name: file_name.to_string(),
                    start_time_ms: start,
                    end_time_ms: end,
                    duration_ms: duration,
                    level1_cluster_id: cluster_id as i32,
                    features: feature_vec,
                });
            }
        }
    }

    // Sort by file and start time
    segments.sort_by(|a, b| {
        a.file_name.cmp(&b.file_name).then_with(|| {
            a.start_time_ms
                .partial_cmp(&b.start_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    Ok(segments)
}

fn load_vocabulary(phase0_dir: &Path) -> Result<Vec<VocabularyEntry>, Box<dyn std::error::Error>> {
    let vocab_path = phase0_dir.join("vocabulary.json");
    let vocab_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&vocab_path)?)?;

    let mut vocabulary = Vec::new();

    if let Some(arr) = vocab_json.as_array() {
        for entry in arr {
            if let (Some(vocab_id), Some(level2_id), Some(count), Some(avg_dur), Some(std_dur)) = (
                entry["vocabulary_id"].as_i64(),
                entry["level2_cluster_id"].as_i64(),
                entry["phrase_count"].as_u64(),
                entry["avg_duration_ms"].as_f64(),
                entry["std_duration_ms"].as_f64(),
            ) {
                vocabulary.push(VocabularyEntry {
                    vocabulary_id: vocab_id as i32,
                    level2_cluster_id: level2_id as i32,
                    phrase_count: count as usize,
                    avg_duration_ms: avg_dur,
                    std_duration_ms: std_dur,
                });
            }
        }
    }

    Ok(vocabulary)
}

// ============================================================================
// Analysis Functions
// ============================================================================

fn analyze_vocabulary_structure(
    vocabulary: &[VocabularyEntry],
    segments: &[Segment],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   📊 Vocabulary Structure:");
    println!();

    // Count Level 2 clusters
    let level2_clusters: HashSet<i32> = vocabulary.iter().map(|v| v.level2_cluster_id).collect();

    println!("      Level 2 clusters: {}", level2_clusters.len());
    println!(
        "      Total phrases in vocabulary: {}",
        vocabulary.iter().map(|v| v.phrase_count).sum::<usize>()
    );
    println!();

    // For each vocabulary entry
    for vocab in vocabulary {
        println!(
            "   Vocabulary {} (Level 2 Cluster {}):",
            vocab.vocabulary_id, vocab.level2_cluster_id
        );
        println!("      Phrase count: {}", vocab.phrase_count);
        println!(
            "      Avg duration: {:.2} ms (±{:.2} ms)",
            vocab.avg_duration_ms, vocab.std_duration_ms
        );
        println!();
    }

    // Check Level 1 distribution
    let level1_counts: HashMap<i32, usize> =
        segments
            .iter()
            .map(|s| s.level1_cluster_id)
            .fold(HashMap::new(), |mut acc, id| {
                *acc.entry(id).or_insert(0) += 1;
                acc
            });

    println!("   📊 Level 1 Cluster Distribution:");
    println!("      Unique Level 1 clusters: {}", level1_counts.len());
    println!();

    let mut sorted_counts: Vec<_> = level1_counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    println!("      Top 20 Level 1 clusters by size:");
    for (i, (cluster_id, count)) in sorted_counts.iter().take(20).enumerate() {
        println!(
            "         {:2}. Cluster {:>3}: {} segments ({:.1}%)",
            i + 1,
            cluster_id,
            count,
            *count as f64 * 100.0 / segments.len() as f64
        );
    }
    println!();

    Ok(())
}

fn detect_position_bias(segments: &[Segment]) -> Result<(), Box<dyn std::error::Error>> {
    println!("   📊 Analyzing Position Bias...");
    println!();

    // Group segments by file
    let mut file_segments: HashMap<String, Vec<&Segment>> = HashMap::new();
    for seg in segments {
        file_segments
            .entry(seg.file_name.clone())
            .or_insert_with(Vec::new)
            .push(seg);
    }

    // For each file, check position vs cluster ID correlation
    let mut position_cluster_pairs: Vec<(usize, i32)> = Vec::new();
    let mut all_sequential = true;
    let mut files_checked = 0;

    for (_file_name, segs) in file_segments.iter().take(1000) {
        // Sort by start time
        let mut sorted: Vec<_> = segs.iter().collect();
        sorted.sort_by(|a, b| a.start_time_ms.partial_cmp(&b.start_time_ms).unwrap());

        // Check if cluster IDs are sequential starting from 0
        let mut expected_id = 0;
        for (pos, seg) in sorted.iter().enumerate() {
            position_cluster_pairs.push((pos, seg.level1_cluster_id));

            if seg.level1_cluster_id != expected_id as i32 {
                all_sequential = false;
            }
            expected_id += 1;
        }

        files_checked += 1;
    }

    // Calculate correlation
    let n = position_cluster_pairs.len();
    let sum_pos: usize = position_cluster_pairs.iter().map(|(p, _)| p).sum();
    let sum_cluster: i32 = position_cluster_pairs.iter().map(|(_, c)| c).sum();
    let sum_pos_cluster: i32 = position_cluster_pairs
        .iter()
        .map(|(p, c)| *c as i32 * *p as i32)
        .sum();

    let mean_pos = sum_pos as f64 / n as f64;
    let mean_cluster = sum_cluster as f64 / n as f64;

    let mut numerator = 0.0;
    let mut sum_sq_diff_pos = 0.0;
    let mut sum_sq_diff_cluster = 0.0;

    for (pos, cluster) in &position_cluster_pairs {
        let diff_pos = *pos as f64 - mean_pos;
        let diff_cluster = *cluster as f64 - mean_cluster;
        numerator += diff_pos * diff_cluster;
        sum_sq_diff_pos += diff_pos * diff_pos;
        sum_sq_diff_cluster += diff_cluster * diff_cluster;
    }

    let correlation = if sum_sq_diff_pos > 0.0 && sum_sq_diff_cluster > 0.0 {
        numerator / (sum_sq_diff_pos.sqrt() * sum_sq_diff_cluster.sqrt())
    } else {
        0.0
    };

    println!("   Results from {} files:", files_checked);
    println!("      Position-Cluster Correlation: {:.4}", correlation);
    println!("      All sequential from 0: {}", all_sequential);
    println!();

    if correlation > 0.95 {
        println!("   ❌ CRITICAL: Near-perfect correlation (>0.95)");
        println!("      → Cluster IDs are DETERMINED by position in file");
        println!("      → This is a CLUSTERING ARTIFACT, not biological structure");
    } else if correlation > 0.5 {
        println!("   ⚠️  WARNING: High correlation (>0.5)");
        println!("      → Position strongly influences cluster assignment");
        println!("      → Clusters may not represent true acoustic types");
    } else {
        println!("   ✅ GOOD: Low correlation (<0.5)");
        println!("      → Cluster assignment is largely position-independent");
        println!("      → Clusters likely represent true acoustic types");
    }
    println!();

    Ok(())
}

fn validate_acoustic_coherence(
    segments: &[Segment],
) -> Result<AcousticAnalysis, Box<dyn std::error::Error>> {
    println!("   📊 Computing Acoustic Distances...");
    println!();

    // Group segments by cluster
    let mut cluster_segments: HashMap<i32, Vec<&Segment>> = HashMap::new();
    for seg in segments {
        cluster_segments
            .entry(seg.level1_cluster_id)
            .or_insert_with(Vec::new)
            .push(seg);
    }

    let feature_dim = segments.first().map(|s| s.features.len()).unwrap_or(0);
    let cluster_count = cluster_segments.len();

    println!("      Feature dimension: {}D", feature_dim);
    println!("      Clusters: {}", cluster_count);
    println!();

    // Sample clusters to analyze (to keep computation manageable)
    let mut cluster_ids: Vec<_> = cluster_segments.keys().copied().collect();
    cluster_ids.sort();

    // Analyze first 20 clusters with sufficient data
    let clusters_to_analyze: Vec<_> = cluster_ids
        .into_iter()
        .filter(|&id| cluster_segments.get(&id).map(|v| v.len()).unwrap_or(0) >= 10)
        .take(20)
        .collect();

    println!(
        "      Analyzing {} clusters with 10+ segments...",
        clusters_to_analyze.len()
    );
    println!();

    let mut within_stats = Vec::new();
    let mut between_distances = Vec::new();

    // Compute within-cluster distances
    for &cluster_id in &clusters_to_analyze {
        if let Some(segs) = cluster_segments.get(&cluster_id) {
            if segs.len() < 2 {
                continue;
            }

            let mut distances = Vec::new();
            // Sample up to 100 pairs
            let sample_size = segs.len().min(100);
            for i in 0..sample_size {
                for j in (i + 1)..sample_size {
                    let dist = euclidean_distance(&segs[i].features, &segs[j].features);
                    distances.push(dist);
                }
            }

            if !distances.is_empty() {
                let mean = distances.iter().sum::<f64>() / distances.len() as f64;
                let variance = distances.iter().map(|&d| (d - mean).powi(2)).sum::<f64>()
                    / distances.len() as f64;
                let std = variance.sqrt();

                within_stats.push(ClusterDistanceStats {
                    cluster_id,
                    segment_count: segs.len(),
                    mean_distance: mean,
                    std_distance: std,
                    min_distance: distances.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    max_distance: distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                });
            }
        }
    }

    // Compute between-cluster distances (sample pairs)
    for i in 0..clusters_to_analyze.len().min(10) {
        for j in (i + 1)..clusters_to_analyze.len().min(10) {
            let cluster_a = clusters_to_analyze[i];
            let cluster_b = clusters_to_analyze[j];

            if let (Some(segs_a), Some(segs_b)) = (
                cluster_segments.get(&cluster_a),
                cluster_segments.get(&cluster_b),
            ) {
                let sample_size = 20.min(segs_a.len()).min(segs_b.len());
                let mut distances = Vec::new();

                for k in 0..sample_size {
                    for l in 0..sample_size {
                        let dist = euclidean_distance(&segs_a[k].features, &segs_b[l].features);
                        distances.push(dist);
                    }
                }

                if !distances.is_empty() {
                    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
                    between_distances.push(BetweenClusterDistance {
                        cluster_a,
                        cluster_b,
                        mean_distance: mean,
                    });
                }
            }
        }
    }

    // Calculate silhouette-like score
    let silhouette = if !within_stats.is_empty() && !between_distances.is_empty() {
        let avg_within =
            within_stats.iter().map(|s| s.mean_distance).sum::<f64>() / within_stats.len() as f64;
        let avg_between = between_distances
            .iter()
            .map(|d| d.mean_distance)
            .sum::<f64>()
            / between_distances.len() as f64;
        (avg_between - avg_within) / avg_between.max(avg_within)
    } else {
        0.0
    };

    // Display results
    println!("   📊 Within-Cluster Distances (Top 10 clusters):");
    println!(
        "   {:<10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Cluster", "Segments", "Mean", "Std", "Min", "Max"
    );
    println!("{}", "-".repeat(75));

    for stat in within_stats.iter().take(10) {
        println!(
            "   {:<10} {:>12} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            stat.cluster_id,
            stat.segment_count,
            stat.mean_distance,
            stat.std_distance,
            stat.min_distance,
            stat.max_distance
        );
    }
    println!();

    println!("   📊 Between-Cluster Distances (Sample):");
    println!(
        "   {:<10} {:<10} {:>12}",
        "Cluster A", "Cluster B", "Mean Distance"
    );
    println!("{}", "-".repeat(40));

    for dist in between_distances.iter().take(15) {
        println!(
            "   {:<10} {:<10} {:>12.4}",
            dist.cluster_a, dist.cluster_b, dist.mean_distance
        );
    }
    println!();

    println!("   📊 Silhouette-like Score: {:.4}", silhouette);
    println!("      (>0.5 = good clustering, <0.2 = poor clustering)");
    println!();

    let coherence = if silhouette > 0.5 {
        "GOOD_CLUSTERS".to_string()
    } else if silhouette > 0.2 {
        "MODERATE_CLUSTERS".to_string()
    } else {
        "POOR_CLUSTERS".to_string()
    };

    Ok(AcousticAnalysis {
        feature_dimension: feature_dim,
        total_segments_analyzed: segments.len(),
        cluster_count,
        within_cluster_distances: within_stats,
        between_cluster_distances: between_distances,
        silhouette_score: silhouette,
        coherence_assessment: coherence,
    })
}

fn analyze_true_transitions(
    segments: &[Segment],
    annotations: &HashMap<String, i32>,
    acoustic_analysis: &AcousticAnalysis,
) -> Result<TransitionAnalysis, Box<dyn std::error::Error>> {
    println!("   📊 Analyzing Acoustic Feature Transitions...");
    println!();

    // Group segments by file
    let mut file_segments: HashMap<String, Vec<&Segment>> = HashMap::new();
    for seg in segments {
        file_segments
            .entry(seg.file_name.clone())
            .or_insert_with(Vec::new)
            .push(seg);
    }

    // For each file, compute feature-based transitions
    let mut context_transitions: HashMap<i32, Vec<FeatureTransition>> = HashMap::new();
    let mut all_transitions = Vec::new();

    for (file_name, segs) in file_segments.iter() {
        let context = annotations.get(file_name).copied();

        // Sort by start time
        let mut sorted: Vec<_> = segs.iter().collect();
        sorted.sort_by(|a, b| a.start_time_ms.partial_cmp(&b.start_time_ms).unwrap());

        // Compute transitions between consecutive segments
        for window in sorted.windows(2) {
            let feature_distance = euclidean_distance(&window[0].features, &window[1].features);
            let duration_ratio = window[1].duration_ms / window[0].duration_ms;

            let transition = FeatureTransition {
                from_cluster: window[0].level1_cluster_id,
                to_cluster: window[1].level1_cluster_id,
                feature_distance,
                duration_ratio,
                context,
            };

            all_transitions.push(transition.clone());

            if let Some(ctx) = context {
                context_transitions
                    .entry(ctx)
                    .or_insert_with(Vec::new)
                    .push(transition);
            }
        }
    }

    // Analyze by context
    let mut context_patterns = HashMap::new();
    let mut entropy_by_context = HashMap::new();

    for (context_id, transitions) in context_transitions {
        // Calculate entropy of feature distances
        let mut distance_bins: Vec<usize> = vec![0; 10];
        for trans in &transitions {
            let bin = (trans.feature_distance * 10.0).floor() as usize;
            let bin = bin.min(9);
            distance_bins[bin] += 1;
        }

        let total = transitions.len() as f64;
        let mut entropy = 0.0;
        for count in distance_bins {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }

        entropy_by_context.insert(context_id, entropy);

        // Find dominant patterns (by feature distance)
        let mut pattern_counts: HashMap<(i32, i32), usize> = HashMap::new();
        for trans in &transitions {
            *pattern_counts
                .entry((trans.from_cluster, trans.to_cluster))
                .or_insert(0) += 1;
        }

        let mut dominant_patterns: Vec<DominantPattern> = pattern_counts
            .into_iter()
            .map(|((from_cluster, to_cluster), count)| DominantPattern {
                cluster_sequence: vec![from_cluster, to_cluster],
                frequency: count,
                proportion: count as f64 / transitions.len() as f64,
            })
            .collect();

        dominant_patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        dominant_patterns.truncate(10);

        let diversity = if dominant_patterns.len() > 1 {
            let proportions: Vec<f64> = dominant_patterns.iter().map(|p| p.proportion).collect();
            calculate_entropy_from_proportions(&proportions)
        } else {
            0.0
        };

        context_patterns.insert(
            context_id,
            ContextPattern {
                context_id,
                num_files: transitions.len(),
                dominant_patterns,
                pattern_diversity: diversity,
            },
        );
    }

    // Calculate cross-context similarity
    let avg_entropy = if !entropy_by_context.is_empty() {
        entropy_by_context.values().sum::<f64>() / entropy_by_context.len() as f64
    } else {
        0.0
    };

    let cross_context_similarity = if entropy_by_context.len() > 1 {
        let entropies: Vec<f64> = entropy_by_context.values().copied().collect();
        let mean = avg_entropy;
        let variance =
            entropies.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / entropies.len() as f64;
        1.0 / (1.0 + variance) // Higher similarity when variance is low
    } else {
        0.0
    };

    // Syntax evidence score (0-1, higher = more evidence for syntax)
    let syntax_evidence_score = if acoustic_analysis.silhouette_score > 0.3 {
        // Good clusters + moderate entropy = potential syntax
        (1.0 - (avg_entropy / 5.0)).max(0.0)
    } else {
        // Poor clusters = likely no meaningful syntax
        0.0
    };

    // Display results
    println!("   📊 Context-Specific Transition Patterns:");
    println!(
        "   {:<10} {:>12} {:>15} {:>15}",
        "Context", "Files", "Diversity", "Dominant Pattern"
    );
    println!("{}", "-".repeat(55));

    let mut sorted_patterns: Vec<_> = context_patterns.values().collect();
    sorted_patterns.sort_by(|a, b| b.num_files.cmp(&a.num_files));

    for pattern in sorted_patterns.iter().take(10) {
        let dominant = pattern
            .dominant_patterns
            .first()
            .map(|p| format!("{:?}", p.cluster_sequence))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "   {:<10} {:>12} {:>15.3} {:>15}",
            pattern.context_id,
            pattern.num_files,
            pattern.pattern_diversity,
            truncate_string(&dominant, 13)
        );
    }
    println!();

    println!("   📊 Transition Entropy by Context:");
    for (ctx, entropy) in entropy_by_context.iter() {
        println!("      Context {}: {:.3} bits", ctx, entropy);
    }
    println!();

    println!(
        "   📊 Cross-Context Similarity: {:.4}",
        cross_context_similarity
    );
    println!("   📊 Syntax Evidence Score: {:.4}", syntax_evidence_score);
    println!();

    Ok(TransitionAnalysis {
        context_specific_patterns: context_patterns,
        cross_context_similarity,
        entropy_by_context,
        syntax_evidence_score,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

#[derive(Debug, Clone)]
struct FeatureTransition {
    from_cluster: i32,
    to_cluster: i32,
    feature_distance: f64,
    duration_ratio: f64,
    context: Option<i32>,
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += (x as f64 - y as f64).powi(2);
    }
    sum.sqrt()
}

fn calculate_entropy_from_proportions(proportions: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &p in proportions {
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn print_recommendations(
    acoustic_analysis: &AcousticAnalysis,
    transition_analysis: &TransitionAnalysis,
) {
    println!(
        "   Based on {} segments analyzed across {} clusters:",
        acoustic_analysis.total_segments_analyzed, acoustic_analysis.cluster_count
    );
    println!();

    // Cluster quality assessment
    println!("   📊 Cluster Quality Assessment:");
    println!(
        "      Silhouette Score: {:.4}",
        acoustic_analysis.silhouette_score
    );
    println!(
        "      Assessment: {}",
        acoustic_analysis.coherence_assessment
    );
    println!();

    if acoustic_analysis.silhouette_score < 0.2 {
        println!("   ❌ CRITICAL: Poor cluster coherence detected");
        println!();
        println!("   RECOMMENDED ACTIONS:");
        println!("   1. Re-cluster using position-independent algorithms:");
        println!("      - Use AgglomerativeClustering with Ward linkage");
        println!("      - Remove temporal/position features from clustering");
        println!("      - Normalize features before clustering");
        println!();
        println!("   2. Validate acoustic features:");
        println!("      - Check if 29D features capture phrase-level semantics");
        println!("      - Consider MFCCs, spectral features, duration");
        println!();
        println!("   3. Alternative approach:");
        println!("      - Use unsupervised deep learning (autoencoders)");
        println!("      - Learn representations directly from audio");
    } else if acoustic_analysis.silhouette_score < 0.5 {
        println!("   ⚠️  MODERATE: Acceptable but improvable cluster quality");
        println!();
        println!("   RECOMMENDED ACTIONS:");
        println!("   1. Fine-tune clustering parameters:");
        println!("      - Adjust min_cluster_size and min_samples");
        println!("      - Try different distance metrics");
        println!();
        println!("   2. Feature engineering:");
        println!("      - Add prosodic features (F0 contour, intensity)");
        println!("      - Consider spectral contrast features");
    } else {
        println!("   ✅ GOOD: Clusters are acoustically coherent");
        println!();
        println!("   NEXT STEPS:");
        println!("   1. Analyze syntax using these validated clusters:");
        println!("      - Build transition matrices");
        println!("      - Test for context-specific patterns");
        println!("      - Calculate predictability metrics");
    }
    println!();

    // Syntax evidence
    println!("   📊 Syntax Evidence Assessment:");
    println!(
        "      Syntax Evidence Score: {:.4}",
        transition_analysis.syntax_evidence_score
    );
    println!(
        "      Cross-Context Similarity: {:.4}",
        transition_analysis.cross_context_similarity
    );
    println!();

    if transition_analysis.syntax_evidence_score > 0.5 {
        println!("   ✅ MODERATE-STRONG evidence for combinatorial syntax");
        println!("      → Proceed with detailed syntactic analysis");
    } else if transition_analysis.syntax_evidence_score > 0.2 {
        println!("   ⚠️  WEAK-MODERATE evidence for combinatorial syntax");
        println!("      → Consider alternative analysis methods");
    } else {
        println!("   ❌ LITTLE to NO evidence for combinatorial syntax");
        println!("      → Egyptian fruit bats likely use context-specific vocabulary");
        println!("      → Not combinatorial syntax like human language");
    }
    println!();
}
