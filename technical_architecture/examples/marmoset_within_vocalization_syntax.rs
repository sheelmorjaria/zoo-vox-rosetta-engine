// Marmoset Within-Vocalization Syntax Discovery
// ================================================
//
// This example implements a comprehensive pipeline to discover syntactic structure
// within marmoset vocalizations by:
//
// 1. **Within-Vocalization Segmentation**: Breaking each audio file into phrase candidates
//    using micro-pauses, intonation shifts, and frequency jumps
//
// 2. **Cross-Vocalization Clustering**: Finding recurring phrases across vocalizations
//
// 3. **Reuse Pattern Analysis**: Distinguishing Grammar/Syntax from Meaning/Content
//    - High Reuse (Across Contexts) = General Purpose Phrases (Grammar/Syntax)
//    - High Specificity (Within Context) = Context Specific Phrases (Meaning/Content)
//
// Usage:
//   cargo run --example marmoset_within_vocalization_syntax --release [vocalizations_dir] [--min-cluster-size N]

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::{
    hdbscan::HdbscanClustering, MicroDynamicsExtractor, WithinVocalizationAnalyzer, WithinVocalizationConfig,
};

/// Marmoset call types (contexts)
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
enum CallType {
    Vocalization,
    Phee,
    Twitter,
    Trill,
    Tsik,
    Seep,
    Infant,
    Unknown,
}

impl CallType {
    fn from_filename(filename: &str) -> Self {
        let fname = filename.to_lowercase();
        if fname.starts_with("vocalization") {
            CallType::Vocalization
        } else if fname.starts_with("phee") {
            CallType::Phee
        } else if fname.starts_with("twitter") {
            CallType::Twitter
        } else if fname.starts_with("trill") {
            CallType::Trill
        } else if fname.starts_with("tsik") {
            CallType::Tsik
        } else if fname.starts_with("seep") {
            CallType::Seep
        } else if fname.starts_with("infant") {
            CallType::Infant
        } else {
            CallType::Unknown
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CallType::Vocalization => "Vocalization",
            CallType::Phee => "Phee",
            CallType::Twitter => "Twitter",
            CallType::Trill => "Trill",
            CallType::Tsik => "Tsik",
            CallType::Seep => "Seep",
            CallType::Infant => "Infant",
            CallType::Unknown => "Unknown",
        }
    }
}

/// Phrase candidate extracted from within a vocalization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseCandidate {
    /// Unique phrase ID (vocalization_file:phrase_index)
    phrase_id: String,
    /// Source vocalization file name
    vocalization_file: String,
    /// Call type of the source vocalization
    call_type: CallType,
    /// Start time within vocalization (ms)
    start_ms: f64,
    /// End time within vocalization (ms)
    end_ms: f64,
    /// Duration (ms)
    duration_ms: f64,
    /// 15D feature vector
    features: Vec<f32>,
    /// F0 value (Hz) if available
    f0_hz: Option<f64>,
    /// Boundary type that started this phrase
    boundary_type: String,
}

/// Vocabulary word (cluster of similar phrase candidates)
#[derive(Debug, Clone)]
struct VocabWord {
    word_id: usize,
    representative_features: Vec<f32>,
    member_phrases: Vec<String>,
    contexts: HashSet<CallType>,
    /// Which vocalizations this word appears in
    source_vocalizations: HashSet<String>,
    /// Total count of occurrences across all vocalizations
    occurrence_count: usize,
}

/// Reuse analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReuseAnalysis {
    /// Words shared across multiple contexts (Grammar/Syntax candidates)
    general_purpose_words: Vec<GeneralPurposeWord>,
    /// Words specific to single context (Meaning/Content candidates)
    context_specific_words: Vec<ContextSpecificWord>,
    /// Statistics per context
    context_statistics: HashMap<String, ContextStatistics>,
}

/// General purpose word (Grammar/Syntax)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeneralPurposeWord {
    word_id: usize,
    num_contexts: usize,
    contexts: Vec<String>,
    occurrence_count: usize,
    num_vocalizations: usize,
}

/// Context specific word (Meaning/Content)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextSpecificWord {
    word_id: usize,
    context: String,
    occurrence_count: usize,
    num_vocalizations: usize,
}

/// Statistics for a single context
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextStatistics {
    context: String,
    total_phrases: usize,
    unique_words: usize,
    general_purpose_words: usize,
    context_specific_words: usize,
    most_frequent_words: Vec<(usize, usize)>, // (word_id, count)
}

/// Sequential pattern analysis results
#[derive(Debug, Clone)]
struct SequentialAnalysis {
    /// Bigram (phrase-pair) statistics
    bigram_counts: HashMap<String, usize>,
    /// Trigram (phrase-triplet) statistics
    trigram_counts: HashMap<String, usize>,
    /// Most common phrase transitions
    top_transitions: Vec<(String, String, usize)>, // (phrase1, phrase2, count)
    /// Sequence patterns per context
    context_sequences: HashMap<String, ContextSequences>,
    /// Cross-context sequence similarities
    cross_context_patterns: Vec<CrossContextPattern>,
}

/// Sequence statistics for a single context
#[derive(Debug, Clone)]
struct ContextSequences {
    context: String,
    num_vocalizations: usize,
    total_sequences: usize,
    avg_sequence_length: f64,
    sequence_lengths: Vec<usize>,
    bigram_entropy: f64,
    most_common_bigrams: Vec<(String, usize)>,
}

/// Pattern shared across contexts
#[derive(Debug, Clone)]
struct CrossContextPattern {
    pattern: Vec<String>, // Sequence of phrase types
    contexts: Vec<String>,
    occurrences: usize,
    pattern_type: String, // "bigram", "trigram", "sequence_motif"
}

/// Progress tracker for long-running operations
struct ProgressTracker {
    total: usize,
    current: Arc<Mutex<usize>>,
    start_time: std::time::Instant,
}

impl ProgressTracker {
    fn new(total: usize) -> Self {
        Self {
            total,
            current: Arc::new(Mutex::new(0)),
            start_time: std::time::Instant::now(),
        }
    }

    fn increment(&self) -> usize {
        let mut current = self.current.lock().unwrap();
        *current += 1;
        let count = *current;

        // Print progress every 1000 items
        if count % 1000 == 0 || count == self.total {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed;
            let remaining = (self.total - count) as f64 / rate;
            println!(
                "  Progress: {}/{} ({:.1}%) | {:.1} items/sec | ETA: {:.1}s",
                count,
                self.total,
                count as f64 / self.total as f64 * 100.0,
                rate,
                remaining
            );
        }

        count
    }
}

/// Recursively find all FLAC files in a directory
fn find_flac_files_recursive(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut flac_files = Vec::new();
    let entries = std::fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectory
            flac_files.extend(find_flac_files_recursive(&path)?);
        } else if path.is_file() {
            // Check if it's a FLAC file
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "flac" {
                    flac_files.push(path);
                }
            }
        }
    }

    Ok(flac_files)
}

/// Load a single FLAC file
fn load_flac_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    audio_samples.extend_from_slice(buf.chan(ch));
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i16::MAX as f32));
                }
            }
            AudioBufferRef::S24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|s| {
                        let raw = s.0 as f32;
                        raw / (i32::MAX as f32 / 256.0)
                    }));
                }
            }
            AudioBufferRef::S32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i32::MAX as f32));
                }
            }
            AudioBufferRef::U8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 128.0) / 128.0));
                }
            }
            _ => return Err("Unsupported audio format".into()),
        }
    }

    Ok(audio_samples)
}

/// Extract 15D features from audio segment
fn extract_15d_features(audio: &[f32], sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features = extractor.extract_15d_marmoset(audio)?;
    Ok(features.to_array().to_vec())
}

/// Segment a vocalization into phrase candidates
fn segment_vocalization(
    audio: &[f32],
    file_name: &str,
    call_type: CallType,
    analyzer: &WithinVocalizationAnalyzer,
    sample_rate: u32,
) -> Result<Vec<PhraseCandidate>, Box<dyn std::error::Error>> {
    // Analyze vocalization for phrase boundaries
    let segmentation = analyzer.analyze_vocalization(audio, None)?;

    let mut phrases = Vec::new();

    // Extract each phrase as a candidate
    for (i, (&start_ms, &duration_ms)) in segmentation
        .phrase_starts_ms
        .iter()
        .zip(segmentation.phrase_durations_ms.iter())
        .enumerate()
    {
        let end_ms = start_ms + duration_ms;

        // Convert time to sample indices
        let start_sample = (start_ms * sample_rate as f64 / 1000.0) as usize;
        let end_sample = (end_ms * sample_rate as f64 / 1000.0) as usize;

        if end_sample > audio.len() || start_sample >= end_sample {
            continue;
        }

        let phrase_audio = &audio[start_sample..end_sample];

        // Skip very short phrases
        if phrase_audio.len() < sample_rate as usize / 100 {
            // Less than 10ms
            continue;
        }

        // Extract 15D features
        let features = extract_15d_features(phrase_audio, sample_rate)?;

        // Get F0 if available
        let f0_hz = segmentation.phrase_f0.get(i).copied();

        let phrase_id = format!("{}:phrase_{}", file_name, i);

        phrases.push(PhraseCandidate {
            phrase_id,
            vocalization_file: file_name.to_string(),
            call_type,
            start_ms,
            end_ms,
            duration_ms,
            features,
            f0_hz,
            boundary_type: if i == 0 {
                "start".to_string()
            } else {
                "detected_boundary".to_string()
            },
        });
    }

    Ok(phrases)
}

/// Discover vocabulary by clustering phrase candidates
fn discover_vocabulary(phrases: &[PhraseCandidate], min_cluster_size: usize, min_samples: usize) -> Vec<VocabWord> {
    if phrases.is_empty() {
        return Vec::new();
    }

    let n_samples = phrases.len();
    let n_features = phrases[0].features.len();

    println!(
        "  📊 Clustering {} phrase candidates ({}D features)...",
        n_samples, n_features
    );
    println!("     ├─ min_cluster_size: {}", min_cluster_size);
    println!("     └─ min_samples: {}", min_samples);

    // For large datasets, process in chunks
    const CHUNK_SIZE: usize = 5000;

    if n_samples <= CHUNK_SIZE {
        return cluster_phrases_single_pass(phrases, min_cluster_size, min_samples);
    }

    println!(
        "  📦 Large dataset, using chunked processing (chunk_size = {})...",
        CHUNK_SIZE
    );

    // Phase 1: Cluster each chunk independently
    let mut chunk_vocabularies: Vec<Vec<VocabWord>> = Vec::new();
    let total_chunks = (n_samples + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (chunk_idx, chunk) in phrases.chunks(CHUNK_SIZE).enumerate() {
        println!(
            "  🔄 Processing chunk {}/{} ({} phrases)...",
            chunk_idx + 1,
            total_chunks,
            chunk.len()
        );

        let chunk_min_cluster_size = ((chunk.len() as f64).sqrt() as usize).max(5);
        let chunk_min_samples = (chunk_min_cluster_size * 3) / 4;

        let chunk_vocabulary = cluster_phrases_single_pass(chunk, chunk_min_cluster_size, chunk_min_samples);

        println!(
            "     → Found {} words in chunk {}",
            chunk_vocabulary.len(),
            chunk_idx + 1
        );
        chunk_vocabularies.push(chunk_vocabulary);
    }

    // Phase 2: Merge vocabularies by clustering their centroids
    println!();
    println!("  🔄 Merging {} chunk vocabularies...", chunk_vocabularies.len());

    let mut all_centroids: Vec<Vec<f32>> = Vec::new();
    let mut all_members: Vec<Vec<String>> = Vec::new();
    let mut all_contexts: Vec<HashSet<CallType>> = Vec::new();
    let mut all_sources: Vec<HashSet<String>> = Vec::new();

    for vocab in &chunk_vocabularies {
        for word in vocab {
            all_centroids.push(word.representative_features.clone());
            all_members.push(word.member_phrases.clone());
            all_contexts.push(word.contexts.clone());
            all_sources.push(word.source_vocalizations.clone());
        }
    }

    if all_centroids.is_empty() {
        return Vec::new();
    }

    // Cluster centroids
    let n_centroids = all_centroids.len();
    let merge_min_cluster_size = ((n_centroids as f64).ln() as usize).max(3);
    let merge_min_samples = (merge_min_cluster_size * 3) / 4;

    println!("     ├─ Clustering {} centroids", n_centroids);
    println!("     ├─ merge_min_cluster_size: {}", merge_min_cluster_size);
    println!("     └─ merge_min_samples: {}", merge_min_samples);

    let mut centroid_matrix = ndarray::Array2::zeros((n_centroids, n_features));
    for (i, centroid) in all_centroids.iter().enumerate() {
        for (j, &val) in centroid.iter().enumerate() {
            centroid_matrix[[i, j]] = val as f64;
        }
    }

    let hdbscan = match HdbscanClustering::new(merge_min_cluster_size, merge_min_samples) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  ⚠ Failed to create HDBSCAN for merging: {:?}", e);
            return create_vocabulary_from_components(all_centroids, all_members, all_contexts, all_sources);
        }
    };

    let labels = match hdbscan.fit_predict(&centroid_matrix) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  ⚠ HDBSCAN merge failed: {:?}", e);
            return create_vocabulary_from_components(all_centroids, all_members, all_contexts, all_sources);
        }
    };

    // Group centroids by merged cluster
    let mut merged_clusters: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            merged_clusters.entry(label).or_insert_with(Vec::new).push(i);
        }
    }

    println!("  ✅ Merged into {} final vocabulary words", merged_clusters.len());

    // Build final vocabulary
    let mut final_vocabulary: Vec<VocabWord> = Vec::new();
    for (merged_id, centroid_indices) in merged_clusters {
        let mut merged_members: Vec<String> = Vec::new();
        let mut merged_contexts: HashSet<CallType> = HashSet::new();
        let mut merged_sources: HashSet<String> = HashSet::new();
        let mut merged_centroid = vec![0.0f32; n_features];

        for &idx in &centroid_indices {
            merged_members.extend(all_members[idx].iter().cloned());
            merged_contexts = merged_contexts.union(&all_contexts[idx]).cloned().collect();
            merged_sources = merged_sources.union(&all_sources[idx]).cloned().collect();
            for (j, &val) in all_centroids[idx].iter().enumerate() {
                merged_centroid[j] += val;
            }
        }

        for val in merged_centroid.iter_mut() {
            *val /= centroid_indices.len() as f32;
        }

        final_vocabulary.push(VocabWord {
            word_id: merged_id as usize,
            representative_features: merged_centroid,
            member_phrases: merged_members,
            contexts: merged_contexts,
            source_vocalizations: merged_sources,
            occurrence_count: all_members.iter().map(|v| v.len()).sum(),
        });
    }

    final_vocabulary.sort_by_key(|w| w.word_id);
    final_vocabulary
}

/// Single-pass clustering for smaller datasets
fn cluster_phrases_single_pass(
    phrases: &[PhraseCandidate],
    min_cluster_size: usize,
    min_samples: usize,
) -> Vec<VocabWord> {
    if phrases.is_empty() {
        return Vec::new();
    }

    let n_samples = phrases.len();
    let n_features = phrases[0].features.len();

    let mut feature_matrix = ndarray::Array2::zeros((n_samples, n_features));
    for (i, phrase) in phrases.iter().enumerate() {
        for (j, &val) in phrase.features.iter().enumerate() {
            feature_matrix[[i, j]] = val as f64;
        }
    }

    let hdbscan = match HdbscanClustering::new(min_cluster_size, min_samples) {
        Ok(h) => h,
        Err(_) => return vec![create_single_word_from_phrases(phrases, 0)],
    };

    let labels = match hdbscan.fit_predict(&feature_matrix) {
        Ok(l) => l,
        Err(_) => return vec![create_single_word_from_phrases(phrases, 0)],
    };

    let mut cluster_map: HashMap<i32, Vec<&PhraseCandidate>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            cluster_map.entry(label).or_insert_with(Vec::new).push(&phrases[i]);
        }
    }

    cluster_map
        .into_iter()
        .map(|(cluster_id, cluster)| {
            let word_id = cluster_id as usize;
            let n_features = cluster[0].features.len();
            let mut centroid = vec![0.0f32; n_features];

            let mut member_phrases: Vec<String> = Vec::new();
            let mut contexts: HashSet<CallType> = HashSet::new();
            let mut source_vocalizations: HashSet<String> = HashSet::new();

            for phrase in &cluster {
                for (i, &val) in phrase.features.iter().enumerate() {
                    centroid[i] += val;
                }
                member_phrases.push(phrase.phrase_id.clone());
                contexts.insert(phrase.call_type);
                source_vocalizations.insert(phrase.vocalization_file.clone());
            }

            for val in centroid.iter_mut() {
                *val /= cluster.len() as f32;
            }

            VocabWord {
                word_id,
                representative_features: centroid,
                member_phrases,
                contexts,
                source_vocalizations,
                occurrence_count: cluster.len(),
            }
        })
        .collect()
}

/// Create vocabulary from components (fallback)
fn create_vocabulary_from_components(
    centroids: Vec<Vec<f32>>,
    members: Vec<Vec<String>>,
    contexts: Vec<HashSet<CallType>>,
    sources: Vec<HashSet<String>>,
) -> Vec<VocabWord> {
    centroids
        .into_iter()
        .zip(members)
        .zip(contexts)
        .zip(sources)
        .enumerate()
        .map(
            |(word_id, (((centroid, member_phrases), contexts), source_vocalizations))| {
                let occurrence_count = member_phrases.len();
                VocabWord {
                    word_id,
                    representative_features: centroid,
                    member_phrases,
                    contexts,
                    source_vocalizations,
                    occurrence_count,
                }
            },
        )
        .collect()
}

/// Create single word cluster (fallback)
fn create_single_word_from_phrases(phrases: &[PhraseCandidate], word_id: usize) -> VocabWord {
    let n_features = phrases[0].features.len();
    let mut centroid = vec![0.0f32; n_features];

    let mut member_phrases: Vec<String> = Vec::new();
    let mut contexts: HashSet<CallType> = HashSet::new();
    let mut source_vocalizations: HashSet<String> = HashSet::new();

    for phrase in phrases {
        for (i, &val) in phrase.features.iter().enumerate() {
            centroid[i] += val;
        }
        member_phrases.push(phrase.phrase_id.clone());
        contexts.insert(phrase.call_type);
        source_vocalizations.insert(phrase.vocalization_file.clone());
    }

    for val in centroid.iter_mut() {
        *val /= phrases.len() as f32;
    }

    VocabWord {
        word_id,
        representative_features: centroid,
        member_phrases,
        contexts,
        source_vocalizations,
        occurrence_count: phrases.len(),
    }
}

/// Analyze phrase reuse patterns
fn analyze_reuse_patterns(vocabulary: &[VocabWord], phrases: &[PhraseCandidate]) -> ReuseAnalysis {
    println!("📊 Analyzing phrase reuse patterns...");

    let mut general_purpose_words = Vec::new();
    let mut context_specific_words = Vec::new();
    let mut context_statistics: HashMap<String, ContextStatistics> = HashMap::new();

    // Count phrases per context
    let mut phrases_per_context: HashMap<CallType, usize> = HashMap::new();
    for phrase in phrases {
        *phrases_per_context.entry(phrase.call_type).or_insert(0) += 1;
    }

    // Classify words and build statistics
    for word in vocabulary {
        let num_contexts = word.contexts.len();
        let num_vocalizations = word.source_vocalizations.len();

        if num_contexts >= 2 {
            // General purpose word (Grammar/Syntax)
            let context_names: Vec<String> = word.contexts.iter().map(|c| c.name().to_string()).collect();

            general_purpose_words.push(GeneralPurposeWord {
                word_id: word.word_id,
                num_contexts,
                contexts: context_names.clone(),
                occurrence_count: word.occurrence_count,
                num_vocalizations,
            });
        } else {
            // Context specific word (Meaning/Content)
            let context_name = word
                .contexts
                .iter()
                .next()
                .map(|c| c.name().to_string())
                .unwrap_or("Unknown".to_string());

            context_specific_words.push(ContextSpecificWord {
                word_id: word.word_id,
                context: context_name.clone(),
                occurrence_count: word.occurrence_count,
                num_vocalizations,
            });
        }
    }

    // Build per-context statistics
    for (call_type, &total_phrases) in &phrases_per_context {
        let context_name = call_type.name();

        let context_words: Vec<_> = vocabulary.iter().filter(|w| w.contexts.contains(call_type)).collect();

        let unique_words = context_words.len();

        let general_purpose = context_words.iter().filter(|w| w.contexts.len() >= 2).count();

        let context_specific = context_words.iter().filter(|w| w.contexts.len() == 1).count();

        // Find most frequent words in this context
        let mut word_freq: Vec<(usize, usize)> =
            context_words.iter().map(|w| (w.word_id, w.occurrence_count)).collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        context_statistics.insert(
            context_name.to_string(),
            ContextStatistics {
                context: context_name.to_string(),
                total_phrases,
                unique_words,
                general_purpose_words: general_purpose,
                context_specific_words: context_specific,
                most_frequent_words: word_freq.into_iter().take(10).collect(),
            },
        );
    }

    // Sort general purpose words by occurrence count
    general_purpose_words.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    // Sort context specific words by occurrence count
    context_specific_words.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    println!("  ✅ Found {} general purpose words", general_purpose_words.len());
    println!("  ✅ Found {} context-specific words", context_specific_words.len());

    ReuseAnalysis {
        general_purpose_words,
        context_specific_words,
        context_statistics,
    }
}

/// Analyze sequential patterns within vocalizations
///
/// This examines how phrases are ordered within each vocalization to discover
/// syntactic rules and patterns.
fn analyze_sequential_patterns(phrases: &[PhraseCandidate]) -> SequentialAnalysis {
    println!("📊 Analyzing sequential patterns within vocalizations...");

    // Group phrases by vocalization (source file)
    let mut vocalization_phrases: HashMap<String, Vec<&PhraseCandidate>> = HashMap::new();
    for phrase in phrases {
        vocalization_phrases
            .entry(phrase.vocalization_file.clone())
            .or_insert_with(Vec::new)
            .push(phrase);
    }

    // Sort phrases within each vocalization by start time
    for phrases_vec in vocalization_phrases.values_mut() {
        phrases_vec.sort_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Extract bigrams and trigrams
    let mut bigram_counts: HashMap<String, usize> = HashMap::new();
    let mut trigram_counts: HashMap<String, usize> = HashMap::new();

    for phrases_vec in vocalization_phrases.values() {
        // Extract bigrams (pairs of consecutive phrases)
        for window in phrases_vec.windows(2) {
            let key = format!("{}→{}", window[0].phrase_id, window[1].phrase_id);
            *bigram_counts.entry(key).or_insert(0) += 1;
        }

        // Extract trigrams (triplets of consecutive phrases)
        for window in phrases_vec.windows(3) {
            let key = format!(
                "{}→{}→{}",
                window[0].phrase_id, window[1].phrase_id, window[2].phrase_id
            );
            *trigram_counts.entry(key).or_insert(0) += 1;
        }
    }

    // Get top transitions
    let mut transition_counts: HashMap<(String, String), usize> = HashMap::new();
    for (bigram, count) in &bigram_counts {
        let parts: Vec<&str> = bigram.split("→").collect();
        if parts.len() == 2 {
            transition_counts.insert((parts[0].to_string(), parts[1].to_string()), *count);
        }
    }

    let mut top_transitions: Vec<(String, String, usize)> = transition_counts
        .into_iter()
        .map(|((p1, p2), count)| (p1, p2, count))
        .collect();
    top_transitions.sort_by(|a, b| b.2.cmp(&a.2));
    top_transitions.truncate(20);

    // Analyze sequences per context
    let mut context_sequences: HashMap<String, ContextSequences> = HashMap::new();

    // Group vocalizations by call type (use HashSet to track unique vocalizations per context)
    let mut context_vocalizations: HashMap<CallType, HashSet<String>> = HashMap::new();
    for phrase in phrases {
        context_vocalizations
            .entry(phrase.call_type)
            .or_insert_with(HashSet::new)
            .insert(phrase.vocalization_file.clone());
    }

    for (call_type, vocalizations) in context_vocalizations {
        let context_name = call_type.name();
        let mut all_bigrams: Vec<String> = Vec::new();
        let mut sequence_lengths: Vec<usize> = Vec::new();

        for vocalization_id in &vocalizations {
            if let Some(phrases_vec) = vocalization_phrases.get(vocalization_id) {
                for window in phrases_vec.windows(2) {
                    let bigram = format!("{}→{}", window[0].phrase_id, window[1].phrase_id);
                    all_bigrams.push(bigram);
                }
                sequence_lengths.push(phrases_vec.len());
            }
        }

        // Calculate bigram entropy
        let bigram_entropy = if all_bigrams.is_empty() {
            0.0
        } else {
            calculate_entropy(&all_bigrams)
        };

        // Count bigram frequencies
        let mut bigram_freq: HashMap<String, usize> = HashMap::new();
        for bigram in &all_bigrams {
            *bigram_freq.entry(bigram.clone()).or_insert(0) += 1;
        }

        let mut most_common_bigrams: Vec<(String, usize)> = bigram_freq.into_iter().collect();
        most_common_bigrams.sort_by(|a, b| b.1.cmp(&a.1));
        most_common_bigrams.truncate(10);

        context_sequences.insert(
            context_name.to_string(),
            ContextSequences {
                context: context_name.to_string(),
                num_vocalizations: vocalizations.len(),
                total_sequences: all_bigrams.len(),
                avg_sequence_length: if sequence_lengths.is_empty() {
                    0.0
                } else {
                    sequence_lengths.iter().map(|&x| x as f64).sum::<f64>() / sequence_lengths.len() as f64
                },
                sequence_lengths,
                bigram_entropy,
                most_common_bigrams,
            },
        );
    }

    // Find cross-context patterns (shared bigrams/trigrams)
    let mut cross_context_patterns: Vec<CrossContextPattern> = Vec::new();

    // Analyze shared bigrams across contexts
    let mut bigram_contexts: HashMap<String, HashSet<CallType>> = HashMap::new();
    for phrase in phrases {
        // Find which bigrams this phrase participates in as first element
        // (simplified - we'd need to track this during bigram extraction for full accuracy)
    }

    // Re-extract with context tracking
    let mut context_bigrams: HashMap<CallType, HashSet<String>> = HashMap::new();
    for phrases_vec in vocalization_phrases.values() {
        if let Some(first_phrase) = phrases_vec.first() {
            let context = first_phrase.call_type;
            for window in phrases_vec.windows(2) {
                let bigram = format!("{}→{}", window[0].phrase_id, window[1].phrase_id);
                context_bigrams
                    .entry(context)
                    .or_insert_with(HashSet::new)
                    .insert(bigram);
            }
        }
    }

    // Find bigrams shared across multiple contexts
    let all_bigrams: HashSet<String> = context_bigrams.values().flat_map(|set| set.iter().cloned()).collect();

    for bigram in all_bigrams {
        let mut contexts: Vec<CallType> = Vec::new();
        for (context, bigrams) in &context_bigrams {
            if bigrams.contains(&bigram) {
                contexts.push(*context);
            }
        }

        if contexts.len() >= 2 {
            cross_context_patterns.push(CrossContextPattern {
                pattern: vec![bigram.clone()],
                contexts: contexts.iter().map(|c| c.name().to_string()).collect(),
                occurrences: bigram_counts.get(&bigram).copied().unwrap_or(0),
                pattern_type: "bigram".to_string(),
            });
        }
    }

    // Sort by occurrences
    cross_context_patterns.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
    cross_context_patterns.truncate(20);

    println!("  ✅ Found {} unique bigrams", bigram_counts.len());
    println!("  ✅ Found {} unique trigrams", trigram_counts.len());
    println!("  ✅ Found {} cross-context patterns", cross_context_patterns.len());

    SequentialAnalysis {
        bigram_counts,
        trigram_counts,
        top_transitions,
        context_sequences,
        cross_context_patterns,
    }
}

/// Calculate entropy of a distribution (measure of unpredictability)
fn calculate_entropy(items: &[String]) -> f64 {
    if items.is_empty() {
        return 0.0;
    }

    let n = items.len();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for item in items {
        *counts.entry(item.as_str()).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / n as f64;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Marmoset Within-Vocalization Syntax Discovery                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut vocalizations_dir = "/home/sheel/birdsong_analysis/data/Vocalizations".to_string();
    let mut min_cluster_size = None;
    let mut limit = None; // Limit number of files to process (for testing)
    let mut skip_clustering = false; // Skip HDBSCAN clustering - treat each phrase as unique

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        limit = Some(n);
                        println!("📊 Limiting to {} files for testing", n);
                        i += 1;
                    }
                }
            }
            "--min-cluster-size" => {
                if i + 1 < args.len() {
                    if let Ok(size) = args[i + 1].parse::<usize>() {
                        min_cluster_size = Some(size);
                        i += 1;
                    }
                }
            }
            "--skip-clustering" => {
                skip_clustering = true;
                println!("⏭ Skip-clustering mode: treating each phrase as its own word");
            }
            arg if i == args.len() - 1 && !arg.starts_with("--") => {
                // Last argument is the directory
                vocalizations_dir = arg.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    // Apply limit if specified
    if let Some(n) = limit {
        println!("📊 Processing limited to {} vocalizations (testing mode)", n);
    }

    let vocalizations_dir = Path::new(&vocalizations_dir);

    if !vocalizations_dir.exists() {
        println!("❌ Vocalizations directory not found: {}", vocalizations_dir.display());
        return Err("Directory not found".into());
    }

    // Sample rate
    let sample_rate = 96000;

    // Configure within-vocalization analyzer for marmoset
    let config = WithinVocalizationConfig {
        sample_rate,
        min_phrase_duration_ms: 15.0, // Minimum 15ms phrases
        min_pause_duration_ms: 8.0,   // Minimum 8ms pauses
        min_f0_change_hz: 1500.0,     // F0 changes > 1.5kHz
        pause_energy_threshold: 0.15, // Energy threshold for pause detection
        frame_size_ms: 5.0,
        hop_size_ms: 2.0,
        require_consensus: false, // Don't require consensus (allow seamless phrases)
        max_phrases: 20,
    };

    let analyzer = WithinVocalizationAnalyzer::new(config);

    println!("📂 Scanning vocalizations directory: {}", vocalizations_dir.display());
    println!();

    // Scan for FLAC files recursively (handles date subdirectories)
    let mut all_files: Vec<(PathBuf, CallType)> = Vec::new();
    let entries = std::fs::read_dir(vocalizations_dir)?;

    // Skip checkpoint directory
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Skip checkpoints and other non-data directories
        if path.ends_with(".checkpoints") || !path.is_dir() {
            continue;
        }

        // Recursively find all FLAC files
        let flac_files = find_flac_files_recursive(&path)?;
        for file_path in flac_files {
            let filename = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            let call_type = CallType::from_filename(filename);
            if call_type != CallType::Unknown {
                all_files.push((file_path, call_type));
            }
        }
    }

    println!("✅ Discovered {} FLAC files", all_files.len());

    // Apply limit if specified
    if let Some(n) = limit {
        let original_len = all_files.len();
        all_files.truncate(n.min(original_len));
        println!("📊 Limited to {} files (was {})", all_files.len(), original_len);
    }

    println!();

    // Phase 1: Segment vocalizations into phrase candidates
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 1: Within-Vocalization Phrase Segmentation                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let progress = ProgressTracker::new(all_files.len());
    let all_phrases: Arc<Mutex<Vec<PhraseCandidate>>> = Arc::new(Mutex::new(Vec::new()));

    // Process in parallel
    all_files.par_iter().for_each(|(file_path, call_type)| {
        let filename = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

        match load_flac_file(file_path) {
            Ok(audio) => match segment_vocalization(&audio, filename, *call_type, &analyzer, sample_rate) {
                Ok(phrases) => {
                    if !phrases.is_empty() {
                        let mut global_phrases = all_phrases.lock().unwrap();
                        global_phrases.extend(phrases);
                    }
                }
                Err(e) => {
                    eprintln!("  Warning: Segmentation failed for {}: {}", filename, e);
                }
            },
            Err(e) => {
                eprintln!("  Warning: Failed to load {}: {}", filename, e);
            }
        }

        progress.increment();
    });

    let phrases = Arc::try_unwrap(all_phrases).unwrap().into_inner()?;
    println!(
        "  ✅ Extracted {} phrase candidates from {} vocalizations",
        phrases.len(),
        all_files.len()
    );
    println!();

    // Phase 2: Discover vocabulary by clustering
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 2: Cross-Vocalization Phrase Clustering                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let vocabulary = if skip_clustering {
        // Skip clustering: treat each phrase as its own word
        println!(
            "  ⏭ Skip-clustering mode: treating each of {} phrases as its own word",
            phrases.len()
        );

        phrases
            .iter()
            .enumerate()
            .map(|(i, phrase)| {
                let mut contexts = HashSet::new();
                contexts.insert(phrase.call_type);

                let mut source_vocalizations = HashSet::new();
                source_vocalizations.insert(phrase.vocalization_file.clone());

                VocabWord {
                    word_id: i,
                    representative_features: phrase.features.clone(),
                    member_phrases: vec![phrase.phrase_id.clone()],
                    contexts,
                    source_vocalizations,
                    occurrence_count: 1,
                }
            })
            .collect()
    } else {
        // Perform HDBSCAN clustering
        let n_total = phrases.len();
        let min_cluster_size_val = min_cluster_size.unwrap_or_else(|| (n_total as f64).sqrt() as usize);
        let min_samples = (min_cluster_size_val * 3) / 4;

        discover_vocabulary(&phrases, min_cluster_size_val, min_samples)
    };

    println!();
    println!("Global Vocabulary Statistics:");
    println!("  Total vocabulary size: {} words", vocabulary.len());
    println!();

    // Phase 3: Analyze reuse patterns
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 3: Phrase Reuse Pattern Analysis                                 │");
    println!("│   (Grammar/Syntax vs Meaning/Content)                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let reuse_analysis = analyze_reuse_patterns(&vocabulary, &phrases);

    // Display results
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    GENERAL PURPOSE WORDS                                ║");
    println!("║              (Grammar/Syntax - Reused Across Contexts)                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Top 20 General Purpose Words (by occurrence count):");
    for (i, word) in reuse_analysis.general_purpose_words.iter().take(20).enumerate() {
        println!(
            "  {:2}. Word {:>4} | {:>2} contexts | {:>4} occurrences | {:>4} vocalizations | {:?}",
            i + 1,
            word.word_id,
            word.num_contexts,
            word.occurrence_count,
            word.num_vocalizations,
            word.contexts
        );
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CONTEXT-SPECIFIC WORDS                               ║");
    println!("║                 (Meaning/Content - Single Context)                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Group context-specific words by context
    let mut context_specific_by_context: HashMap<String, Vec<&ContextSpecificWord>> = HashMap::new();
    for word in &reuse_analysis.context_specific_words {
        context_specific_by_context
            .entry(word.context.clone())
            .or_insert_with(Vec::new)
            .push(word);
    }

    for (context, words) in context_specific_by_context.iter() {
        println!("{}: {} context-specific words", context, words.len());
        let mut sorted_words = words.clone();
        sorted_words.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

        for (i, word) in sorted_words.iter().take(5).enumerate() {
            println!(
                "  {:2}. Word {:>4} | {:>4} occurrences | {:>4} vocalizations",
                i + 1,
                word.word_id,
                word.occurrence_count,
                word.num_vocalizations
            );
        }
        println!();
    }

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CONTEXT STATISTICS                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Context", "Phrases", "Unique", "General", "Specific", "GP%"
    );
    println!("{}", "-".repeat(92));

    for (context_name, stats) in &reuse_analysis.context_statistics {
        let gp_pct = if stats.unique_words > 0 {
            stats.general_purpose_words as f64 / stats.unique_words as f64 * 100.0
        } else {
            0.0
        };

        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>11.1}%",
            context_name,
            stats.total_phrases,
            stats.unique_words,
            stats.general_purpose_words,
            stats.context_specific_words,
            gp_pct
        );
    }

    // Phase 4: Sequential pattern analysis
    let _sequential_analysis = run_sequential_analysis(&phrases)?;

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Analysis complete!");
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

// ============================================================================
// PHASE 4: SEQUENTIAL PATTERN ANALYSIS
// ============================================================================

fn run_sequential_analysis(phrases: &[PhraseCandidate]) -> Result<SequentialAnalysis, Box<dyn std::error::Error>> {
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 4: Sequential Pattern Analysis                                   │");
    println!("│   (Discovering Syntactic Rules Through Phrase Ordering)                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let analysis = analyze_sequential_patterns(phrases);

    // Display sequential pattern results
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    TOP SEQUENTIAL TRANSITIONS                           ║");
    println!("║              (Most Common Phrase-to-Phrase Patterns)                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Top 20 Bigram Transitions:");
    for (i, (from_phrase, to_phrase, count)) in analysis.top_transitions.iter().take(20).enumerate() {
        println!(
            "  {:2}. {:30} → {:30} ({:>3} occurrences)",
            i + 1,
            truncate_phrase_id(from_phrase),
            truncate_phrase_id(to_phrase),
            count
        );
    }

    println!();
    println!("Bigram Statistics:");
    println!("  Total unique bigrams: {}", analysis.bigram_counts.len());
    println!("  Total unique trigrams: {}", analysis.trigram_counts.len());

    // Calculate overall statistics
    let total_bigram_count: usize = analysis.bigram_counts.values().sum();
    let avg_bigram_count = if !analysis.bigram_counts.is_empty() {
        total_bigram_count as f64 / analysis.bigram_counts.len() as f64
    } else {
        0.0
    };
    println!("  Total bigram occurrences: {}", total_bigram_count);
    println!("  Average occurrences per bigram: {:.2}", avg_bigram_count);

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PER-CONTEXT SEQUENCE PATTERNS                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Sort contexts by sequence count
    let mut context_entries: Vec<_> = analysis.context_sequences.iter().collect();
    context_entries.sort_by(|a, b| b.1.total_sequences.cmp(&a.1.total_sequences));

    println!(
        "{:<25} {:>12} {:>12} {:>15} {:>12} {:>15}",
        "Context", "Vocalizations", "Sequences", "Avg Length", "Bigram Entropy", "Top Bigram"
    );
    println!("{}", "-".repeat(110));

    for (context_name, seq_data) in context_entries.iter().take(10) {
        let top_bigram = seq_data
            .most_common_bigrams
            .first()
            .map(|(bigram, _)| truncate_string(bigram, 20))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<25} {:>12} {:>12} {:>15.2} {:>12.3} {:>15}",
            context_name,
            seq_data.num_vocalizations,
            seq_data.total_sequences,
            seq_data.avg_sequence_length,
            seq_data.bigram_entropy,
            top_bigram
        );
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CROSS-CONTEXT SHARED PATTERNS                        ║");
    println!("║              (Syntactic Rules Shared Across Contexts)                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    if analysis.cross_context_patterns.is_empty() {
        println!("  No cross-context shared patterns found.");
        println!("  (This suggests each context has unique phrase ordering rules)");
    } else {
        println!("Top 15 Cross-Context Shared Patterns:");
        for (i, pattern) in analysis.cross_context_patterns.iter().take(15).enumerate() {
            let pattern_str = pattern.pattern.join(" → ");
            println!("  {:2}. Pattern: {}", i + 1, truncate_string(&pattern_str, 50));
            println!(
                "      Shared across {} contexts: {:?}",
                pattern.contexts.len(),
                pattern.contexts.iter().take(3).cloned().collect::<Vec<_>>()
            );
            println!("      Total occurrences: {}", pattern.occurrences);
            println!();
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Sequential Pattern Analysis Summary:");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Calculate overall entropy across all contexts
    let all_entropies: Vec<f64> = analysis.context_sequences.values().map(|s| s.bigram_entropy).collect();

    if !all_entropies.is_empty() {
        let avg_entropy = all_entropies.iter().sum::<f64>() / all_entropies.len() as f64;
        let min_entropy = all_entropies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_entropy = all_entropies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!("Average bigram entropy (unpredictability): {:.3} bits", avg_entropy);
        println!("Entropy range: [{:.3}, {:.3}] bits", min_entropy, max_entropy);
        println!();

        // Interpret entropy
        if avg_entropy < 2.0 {
            println!("→ LOW ENTROPY: Highly predictable phrase sequences");
            println!("  Suggests rigid syntactic rules with common phrase transitions");
        } else if avg_entropy < 4.0 {
            println!("→ MEDIUM ENTROPY: Moderately predictable phrase sequences");
            println!("  Suggests flexible syntax with some common patterns");
        } else {
            println!("→ HIGH ENTROPY: Highly unpredictable phrase sequences");
            println!("  Suggests free-form syntax with many possible transitions");
        }
    }

    println!();
    println!("Cross-context patterns: {}", analysis.cross_context_patterns.len());
    if !analysis.cross_context_patterns.is_empty() {
        let max_shared = analysis
            .cross_context_patterns
            .iter()
            .map(|p| p.contexts.len())
            .max()
            .unwrap_or(0);
        println!("Maximum pattern sharing: {} contexts", max_shared);
    }

    Ok(analysis)
}

fn truncate_phrase_id(id: &str) -> String {
    // Extract just the phrase index from the full ID
    // Format: "phee_20200101_120000_0 phrase_00123" -> "phrase_00123"
    if let Some(space_idx) = id.find(' ') {
        id[space_idx + 1..].to_string()
    } else {
        id.to_string()
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
