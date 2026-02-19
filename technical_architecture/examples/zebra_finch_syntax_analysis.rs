//! Syntax Analysis: Markov Chains for Phrase Sequences
//!
//! Analyzes the "rules of combination" for discovered atomic phrases.
//! Builds transition matrices (Bigram/Trigram probabilities) to understand
//! the syntax of zebra finch vocalizations.
//!
//! Usage:
//!   cargo run --release --example zebra_finch_syntax_analysis

use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    AcousticSimilarityEngine, DynamicPhraseCandidate, DynamicSegmenter, DynamicSegmenterConfig,
    SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;
const SAMPLE_RATE: u32 = 44100;

// Type alias for phrase with metadata
type PhraseWithMeta = (DynamicPhraseCandidate, String, String);

// ============================================================================
// SYNTAX MODELS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BigramModel {
    transitions: HashMap<(usize, usize), usize>,
    from_totals: HashMap<usize, usize>,
    vocabulary: HashSet<usize>,
}

impl BigramModel {
    fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            from_totals: HashMap::new(),
            vocabulary: HashSet::new(),
        }
    }

    fn add_transition(&mut self, from: usize, to: usize) {
        *self.transitions.entry((from, to)).or_insert(0) += 1;
        *self.from_totals.entry(from).or_insert(0) += 1;
        self.vocabulary.insert(from);
        self.vocabulary.insert(to);
    }

    fn probability(&self, from: usize, to: usize) -> f64 {
        let total = self.from_totals.get(&from).copied().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        self.transitions.get(&(from, to)).copied().unwrap_or(0) as f64 / total as f64
    }

    fn perplexity(&self, sequences: &[Vec<usize>]) -> f64 {
        let mut log_prob = 0.0;
        let mut total = 0;

        for seq in sequences {
            for window in seq.windows(2) {
                let prob = self.probability(window[0], window[1]);
                if prob > 0.0 {
                    log_prob += prob.ln();
                    total += 1;
                }
            }
        }

        if total == 0 {
            return f64::INFINITY;
        }

        let avg_log_prob = log_prob / total as f64;
        (-avg_log_prob).exp()
    }
}

// ============================================================================
// SYNTAX ANALYSIS REPORT
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SyntaxAnalysisReport {
    species: String,
    total_sequences: usize,
    vocabulary_size: usize,
    bigram_stats: BigramStats,
    bird_syntax_comparison: Vec<BirdSyntaxProfile>,
    common_patterns: Vec<PatternInfo>,
    processing_time_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BigramStats {
    unique_transitions: usize,
    entropy: f64,
    perplexity: f64,
    top_transitions: Vec<TransitionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransitionInfo {
    from_phrase: usize,
    to_phrase: usize,
    count: usize,
    probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BirdSyntaxProfile {
    bird_id: String,
    sequence_count: usize,
    unique_phrases: usize,
    perplexity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternInfo {
    pattern: Vec<usize>,
    occurrences: usize,
}

// ============================================================================
// ANNOTATION
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    #[serde(rename = "fn")]
    filename: String,
    call_type: String,
    name: String,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Syntax Analysis: Markov Chains for Phrase Sequences                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    let data_dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join("birdsong_analysis/data/zebra_finch/zebra_finch");
    let vocalizations_dir = data_dir.join("vocalizations");
    let annotations_path = data_dir.join("annotations.csv");

    let segmenter_config = DynamicSegmenterConfig::zebra_finch();
    let segmenter = DynamicSegmenter::new(segmenter_config.clone(), SAMPLE_RATE);

    let annotations = load_annotations(&annotations_path)?;
    let max_files = 500;
    let annotations_subset: Vec<_> = annotations.into_iter().take(max_files).collect();
    println!("Processing {} vocalizations...", max_files);

    // ========================================================================
    // Extract phrase sequences
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[1/3] Extracting Phrase Sequences");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let processed = Arc::new(AtomicUsize::new(0));

    let all_candidates: Vec<PhraseWithMeta> = annotations_subset
        .par_iter()
        .flat_map(|ann| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                println!("  Progress: {}/{}", count + 1, max_files);
            }

            let audio_path = vocalizations_dir.join(&ann.filename);
            if let Ok(audio) = load_audio(&audio_path) {
                if audio.len() < 500 {
                    return Vec::new();
                }

                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(
                    SAMPLE_RATE,
                )));
                let result = segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64)
                            .ok()
                            .map(|f| f.to_vector().to_vec())
                    },
                    &ann.filename,
                );

                let mut cands: Vec<PhraseWithMeta> = result
                    .candidates
                    .into_iter()
                    .map(|c| (c, ann.call_type.clone(), ann.name.clone()))
                    .collect();
                cands.sort_by(|a, b| a.0.start_ms.partial_cmp(&b.0.start_ms).unwrap());
                cands
            } else {
                Vec::new()
            }
        })
        .collect();

    // Group by source file
    let mut file_sequences: HashMap<String, Vec<PhraseWithMeta>> = HashMap::new();
    for item in all_candidates {
        file_sequences
            .entry(item.0.source_file.clone())
            .or_insert_with(Vec::new)
            .push(item);
    }

    // Sort each sequence by start time
    for seq in file_sequences.values_mut() {
        seq.sort_by(|a, b| a.0.start_ms.partial_cmp(&b.0.start_ms).unwrap());
    }

    println!(
        "\nExtracted {} sequences from {} files",
        file_sequences.len(),
        max_files
    );

    // ========================================================================
    // Cluster phrases to get phrase IDs
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/3] Clustering Phrases to Build Vocabulary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Collect all candidates for clustering
    let flat_candidates: Vec<(DynamicPhraseCandidate, String)> = file_sequences
        .values()
        .flat_map(|seq| seq.iter().map(|(c, ct, _b)| (c.clone(), ct.clone())))
        .collect();

    let phrase_clusters = cluster_phrases(&flat_candidates, 0.30, 2);
    println!("Discovered {} phrase types", phrase_clusters.len());

    // Build phrase ID lookup
    let mut phrase_to_id: HashMap<String, usize> = HashMap::new();
    for cluster in &phrase_clusters {
        for &idx in &cluster.member_indices {
            if let Some((cand, _)) = flat_candidates.get(idx) {
                phrase_to_id
                    .entry(cand.id.clone())
                    .or_insert(cluster.phrase_id);
            }
        }
    }

    // Convert sequences to phrase IDs
    let mut sequences: Vec<Vec<usize>> = Vec::new();
    let mut bird_sequences: HashMap<String, Vec<Vec<usize>>> = HashMap::new();

    for seq in file_sequences.values() {
        let phrase_seq: Vec<usize> = seq
            .iter()
            .filter_map(|(cand, _, _)| phrase_to_id.get(&cand.id).copied())
            .collect();

        if phrase_seq.len() >= 2 {
            sequences.push(phrase_seq.clone());

            if let Some((_, _, bird)) = seq.first() {
                bird_sequences
                    .entry(bird.clone())
                    .or_insert_with(Vec::new)
                    .push(phrase_seq);
            }
        }
    }

    println!("Built {} phrase sequences", sequences.len());

    // ========================================================================
    // Build Markov models
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/3] Building Markov Models (Bigram)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut bigram = BigramModel::new();
    for seq in &sequences {
        for window in seq.windows(2) {
            bigram.add_transition(window[0], window[1]);
        }
    }

    println!(
        "Bigram model: {} unique transitions",
        bigram.transitions.len()
    );

    let perplexity = bigram.perplexity(&sequences);
    let entropy = calculate_entropy(&bigram);

    println!("\nSyntax Statistics:");
    println!("  ├─ Vocabulary Size: {} phrases", bigram.vocabulary.len());
    println!("  ├─ Unique Transitions: {}", bigram.transitions.len());
    println!("  ├─ Entropy: {:.3} bits", entropy);
    println!(
        "  └─ Perplexity: {:.2} (lower = more predictable)",
        perplexity
    );

    // ========================================================================
    // Find common patterns
    // ========================================================================
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    for seq in &sequences {
        for window in seq.windows(3) {
            *pattern_counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }

    let mut common_patterns: Vec<PatternInfo> = pattern_counts
        .iter()
        .filter(|(_, &count)| count >= 3)
        .map(|(pattern, &count)| PatternInfo {
            pattern: pattern.clone(),
            occurrences: count,
        })
        .collect();
    common_patterns.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));

    // ========================================================================
    // Per-bird syntax comparison
    // ========================================================================
    let mut bird_profiles: Vec<BirdSyntaxProfile> = Vec::new();
    for (bird, bird_seqs) in &bird_sequences {
        if bird_seqs.len() < 5 {
            continue;
        }

        let mut bird_bigram = BigramModel::new();
        for seq in bird_seqs {
            for window in seq.windows(2) {
                bird_bigram.add_transition(window[0], window[1]);
            }
        }

        bird_profiles.push(BirdSyntaxProfile {
            bird_id: bird.clone(),
            sequence_count: bird_seqs.len(),
            unique_phrases: bird_bigram.vocabulary.len(),
            perplexity: bird_bigram.perplexity(bird_seqs),
        });
    }

    // ========================================================================
    // Generate report
    // ========================================================================
    let total_time = total_start.elapsed();

    let mut top_transitions: Vec<TransitionInfo> = bigram
        .transitions
        .iter()
        .map(|((from, to), &count)| TransitionInfo {
            from_phrase: *from,
            to_phrase: *to,
            count,
            probability: bigram.probability(*from, *to),
        })
        .collect();
    top_transitions.sort_by(|a, b| b.count.cmp(&a.count));

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SYNTAX ANALYSIS SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("TOP 5 BIGRAM TRANSITIONS (Most Common Phrase Sequences):");
    for (i, t) in top_transitions.iter().take(5).enumerate() {
        println!(
            "  {}. Phrase {} → Phrase {}: {} occurrences ({:.1}%)",
            i + 1,
            t.from_phrase,
            t.to_phrase,
            t.count,
            t.probability * 100.0
        );
    }

    println!("\nTOP 5 COMMON PATTERNS (Repeated 3-phrase sequences):");
    for (i, p) in common_patterns.iter().take(5).enumerate() {
        println!(
            "  {}. [{} → {} → {}]: {} occurrences",
            i + 1,
            p.pattern[0],
            p.pattern[1],
            p.pattern[2],
            p.occurrences
        );
    }

    println!("\nPER-BIRD SYNTAX VARIATION:");
    for profile in bird_profiles.iter().take(5) {
        println!(
            "  Bird {}: {} sequences, {} phrases, perplexity: {:.2}",
            profile.bird_id, profile.sequence_count, profile.unique_phrases, profile.perplexity
        );
    }

    let report = SyntaxAnalysisReport {
        species: "zebra_finch".to_string(),
        total_sequences: sequences.len(),
        vocabulary_size: bigram.vocabulary.len(),
        bigram_stats: BigramStats {
            unique_transitions: bigram.transitions.len(),
            entropy,
            perplexity,
            top_transitions: top_transitions.into_iter().take(20).collect(),
        },
        bird_syntax_comparison: bird_profiles,
        common_patterns: common_patterns.into_iter().take(20).collect(),
        processing_time_sec: total_time.as_secs_f64(),
    };

    std::fs::create_dir_all("zebra_finch_analysis")?;
    let output_path = "zebra_finch_analysis/syntax_analysis.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report)?;

    println!("\nReport saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn calculate_entropy(bigram: &BigramModel) -> f64 {
    let mut entropy = 0.0;

    for from in &bigram.vocabulary {
        let total = bigram.from_totals.get(from).copied().unwrap_or(0) as f64;
        if total == 0.0 {
            continue;
        }

        for to in &bigram.vocabulary {
            let prob = bigram.probability(*from, *to);
            if prob > 0.0 {
                entropy -= prob * prob.log2();
            }
        }
    }

    entropy / bigram.vocabulary.len() as f64
}

struct PhraseCluster {
    phrase_id: usize,
    member_indices: Vec<usize>,
}

fn cluster_phrases(
    candidates: &[(DynamicPhraseCandidate, String)],
    threshold: f32,
    min_size: usize,
) -> Vec<PhraseCluster> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    let n_samples = candidates.len().min(5000);
    let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
    for (i, (cand, _)) in candidates.iter().take(n_samples).enumerate() {
        for (j, &val) in cand.features.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }
    engine.fit_normalization(&matrix);

    let mut clusters: Vec<PhraseCluster> = Vec::new();
    let mut assigned = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster_indices = vec![i];
        assigned[i] = true;

        let query = Array1::from_vec(candidates[i].0.features.clone());

        for j in (i + 1)..candidates.len() {
            if !assigned[j] {
                let candidate = Array1::from_vec(candidates[j].0.features.clone());
                let dist = engine.distance(&query, &candidate);

                if dist < threshold as f64 {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        if cluster_indices.len() >= min_size {
            clusters.push(PhraseCluster {
                phrase_id: clusters.len(),
                member_indices: cluster_indices,
            });
        }
    }

    clusters
}

fn load_annotations(path: &Path) -> Result<Vec<Annotation>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut annotations = Vec::new();
    for result in csv_reader.deserialize() {
        let annotation: Annotation = result?;
        annotations.push(annotation);
    }

    Ok(annotations)
}

fn load_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(audio)
}
