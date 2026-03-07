// Phase 0: Level 2 Reclustering with Cosine Similarity
//
// This script reuses existing segments from all_segments.json and reclusters
// them using cosine similarity instead of Euclidean distance.
//
// Usage: cargo run --release --example phase0_recluster_cosine

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;
use technical_architecture::hdbscan::{DistanceMetric, HdbscanClustering};

const LEVEL2_MIN_CLUSTER_SIZE: usize = 5;
const LEVEL2_MIN_SAMPLES: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseSegment {
    segment_id: usize,
    file_index: usize,
    file_name: String,
    start_time_ms: f64,
    end_time_ms: f64,
    duration_ms: f64,
    start_sample: usize,
    end_sample: usize,
    sample_rate: u32,
    frame_indices: Vec<usize>,
    level1_cluster_id: i32,
    representative_features: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VocabularyItem {
    vocabulary_id: usize,
    level2_cluster_id: i32,
    phrase_count: usize,
    avg_duration_ms: f64,
    std_duration_ms: f64,
    example_phrases: Vec<PhraseSegment>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Phase 0: Level 2 Reclustering with COSINE SIMILARITY                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Reuses existing segments from all_segments.json                         ║");
    println!("║  Re-runs Level 2 HDBSCAN with cosine distance metric                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let results_dir = data_dir.join("phase0_twolevel_hdbscan_cosine_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load existing segments
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Pre-Extracted Segments                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let segments_path = data_dir.join("phase0_twolevel_hdbscan_results/all_segments.json");
    println!("   📂 Loading from: {}", segments_path.display());

    let load_start = Instant::now();
    let segments_json = fs::read_to_string(&segments_path)?;
    let all_segments: Vec<PhraseSegment> = serde_json::from_str(&segments_json)?;
    let load_time = load_start.elapsed();

    println!(
        "      ✅ Loaded {} segments in {:.2}s",
        all_segments.len(),
        load_time.as_secs_f64()
    );
    println!(
        "      ├─ Feature dimensions: {}D",
        all_segments[0].representative_features.len()
    );
    println!(
        "      └─ Total files: {}",
        all_segments
            .iter()
            .map(|s| &s.file_name)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!();

    // ========================================================================
    // Step 2: Build Level 2 vocabulary with COSINE distance
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Vocabulary with COSINE Similarity                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   🔧 HDBSCAN Configuration:");
    println!("      ├─ min_cluster_size: {}", LEVEL2_MIN_CLUSTER_SIZE);
    println!("      ├─ min_samples: {}", LEVEL2_MIN_SAMPLES);
    println!("      └─ Distance metric: COSINE (scale-invariant, pattern-based)");
    println!();

    let hdbscan = HdbscanClustering::with_metric(LEVEL2_MIN_CLUSTER_SIZE, LEVEL2_MIN_SAMPLES, DistanceMetric::Cosine)?;

    println!("   🔍 Running Level 2 HDBSCAN...");
    let vocab_start = Instant::now();

    // Extract features matrix
    let n_segments = all_segments.len();
    let n_features = all_segments[0].representative_features.len();

    let mut flat_features = Vec::with_capacity(n_segments * n_features);
    for segment in &all_segments {
        flat_features.extend_from_slice(&segment.representative_features);
    }

    let features_array = Array2::from_shape_vec((n_segments, n_features), flat_features)?;

    // Run HDBSCAN
    let labels = hdbscan.fit_predict_hnsw(&features_array)?;
    let vocab_time = vocab_start.elapsed();

    println!("      ✅ Clustering completed in {:.2}s", vocab_time.as_secs_f64());

    // Group segments by cluster
    let mut cluster_map: std::collections::HashMap<i32, Vec<&PhraseSegment>> = std::collections::HashMap::new();
    for (segment_idx, &label) in labels.iter().enumerate() {
        if label >= 0 {
            cluster_map
                .entry(label)
                .or_insert_with(Vec::new)
                .push(&all_segments[segment_idx]);
        }
    }

    let mut vocabulary = Vec::new();
    let mut vocab_id = 0;

    let mut cluster_ids: Vec<_> = cluster_map.keys().cloned().collect();
    cluster_ids.sort();

    for &cluster_id in &cluster_ids {
        let segments = cluster_map.get(&cluster_id).unwrap();

        let durations: Vec<f64> = segments.iter().map(|s| s.duration_ms).collect();
        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance = durations.iter().map(|&d| (d - avg_duration).powi(2)).sum::<f64>() / durations.len() as f64;
        let std_duration = variance.sqrt();

        vocabulary.push(VocabularyItem {
            vocabulary_id: vocab_id,
            level2_cluster_id: cluster_id,
            phrase_count: segments.len(),
            avg_duration_ms: avg_duration,
            std_duration_ms: std_duration,
            example_phrases: segments.iter().map(|&s| s.clone()).collect(),
        });

        vocab_id += 1;
    }

    println!("      📚 Discovered {} vocabulary items", vocabulary.len());
    println!();

    // ========================================================================
    // Step 3: Display Statistics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Vocabulary Statistics (COSINE vs Previous EUCLIDEAN)                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut sorted_vocab: Vec<_> = vocabulary.iter().collect();
    sorted_vocab.sort_by(|a, b| b.phrase_count.cmp(&a.phrase_count));

    println!("   🎯 Top 20 Vocabulary Items:");
    println!("      ┌──────┬──────────────┬──────────────┬─────────────┬────────────┐");
    println!("      │  ID  │   Phrases    │  Avg Dur(ms) │ Std Dur(ms) │ Type       │");
    println!("      ├──────┼──────────────┼──────────────┼─────────────┼────────────┤");

    for item in sorted_vocab.iter().take(20) {
        let vocab_type = if item.phrase_count > 100 {
            "VERY_COMMON"
        } else if item.phrase_count > 50 {
            "COMMON"
        } else if item.phrase_count > 20 {
            "MODERATE"
        } else {
            "RARE"
        };

        println!(
            "      │ {:4} │ {:12} │ {:12.1} │ {:11.1} │ {:10} │",
            item.vocabulary_id, item.phrase_count, item.avg_duration_ms, item.std_duration_ms, vocab_type
        );
    }

    println!("      └──────┴──────────────┴──────────────┴─────────────┴────────────┘");
    println!();

    // ========================================================================
    // Step 4: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Saving Cosine-Based Results                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save vocabulary
    let vocab_path = results_dir.join("vocabulary.json");
    let vocab_json = serde_json::to_string_pretty(&vocabulary)?;
    fs::write(&vocab_path, vocab_json)?;
    println!("   💾 Vocabulary: {}", vocab_path.display());

    // Save all segments (same as input, but with updated context)
    let segments_path = results_dir.join("all_segments.json");
    let segments_json = serde_json::to_string_pretty(&all_segments)?;
    fs::write(&segments_path, segments_json)?;
    println!("   💾 All segments: {}", segments_path.display());

    // Save timestamp map
    let timestamp_map_path = results_dir.join("timestamp_map.json");
    let timestamp_map: Vec<serde_json::Value> = all_segments
        .iter()
        .map(|seg| {
            serde_json::json!({
                "file_name": seg.file_name,
                "segment_id": seg.segment_id,
                "vocabulary_id": seg.level1_cluster_id,
                "start_time_ms": seg.start_time_ms,
                "end_time_ms": seg.end_time_ms,
                "duration_ms": seg.duration_ms,
                "start_sample": seg.start_sample,
                "end_sample": seg.end_sample,
                "sample_rate": seg.sample_rate,
            })
        })
        .collect();
    let timestamp_json = serde_json::to_string_pretty(&timestamp_map)?;
    fs::write(&timestamp_map_path, timestamp_json)?;
    println!("   💾 Timestamp map: {}", timestamp_map_path.display());

    println!();

    // ========================================================================
    // Step 5: Generate Symbolic Stream
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Symbolic Stream Generation (COSINE-Based Vocabulary)                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut symbolic_stream: Vec<i32> = Vec::new();
    let mut segment_to_vocab: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();

    for vocab in &vocabulary {
        for phrase in &vocab.example_phrases {
            segment_to_vocab.insert(phrase.segment_id, vocab.vocabulary_id as i32);
        }
    }

    let mut sorted_segments: Vec<_> = all_segments.iter().collect();
    sorted_segments.sort_by(|a, b| {
        a.file_index
            .cmp(&b.file_index)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });

    for segment in sorted_segments {
        let vocab_id = segment_to_vocab.get(&segment.segment_id).copied().unwrap_or(-1);
        symbolic_stream.push(vocab_id);
    }

    println!("   📝 Symbolic Stream Statistics:");
    println!("      ├─ Total symbols: {}", symbolic_stream.len());
    println!("      ├─ Unique vocabulary items: {}", vocabulary.len());
    println!(
        "      └─ Noise symbols: {}",
        symbolic_stream.iter().filter(|&&x| x == -1).count()
    );

    let stream_path = results_dir.join("symbolic_stream.txt");
    let stream_text: String = symbolic_stream
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&stream_path, stream_text)?;
    println!("   💾 Symbolic stream: {}", stream_path.display());

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    RECLUSTERING COMPLETE                                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  SUMMARY:                                                                 ║");
    println!(
        "║     • Segments reused: {}                                              ║",
        all_segments.len()
    );
    println!(
        "║     • Vocabulary items: {}                                               ║",
        vocabulary.len()
    );
    println!("║     • Distance metric: COSINE (pattern-based)                            ║");
    println!(
        "║     • Results directory: {}                               ║",
        results_dir.display()
    );
    println!("║                                                                           ║");
    println!("║  NEXT: Run phrase_context_analysis_bat_generality with new vocabulary     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
