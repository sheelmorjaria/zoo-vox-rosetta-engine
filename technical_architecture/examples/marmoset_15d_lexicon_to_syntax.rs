// Marmoset Lexicon-to-Syntax Analysis (15D RFE-Optimized Features)
// ==============================================================
//
// This example demonstrates cross-context lexicon discovery and syntactic analysis
// for marmoset vocalizations using the 15D RFE-optimized feature set.
//
// **Research Questions:**
// 1. What is the "vocabulary" (lexicon) for each marmoset call type?
// 2. How does syntactic structure vary across call type contexts?
// 3. Are there universal words vs. context-specific words?
// 4. Does combinatorial structure exist within and across call types?
//
// **Features Used:**
// 15D RFE-optimized features specifically selected for marmoset call type discrimination
// via Recursive Feature Elimination (RFE) using Fisher scores.
//
// **Clustering:**
// Uses HDBSCAN (Hierarchical Density-Based Spatial Clustering) for vocabulary discovery.
// HDBSCAN automatically determines the number of clusters and handles variable density.
//
// **Checkpointing:**
// - Progress is automatically saved after each call type is processed
// - Use --resume flag to continue from checkpoint
// - Checkpoint files saved to <vocalizations_dir>/.checkpoints/
//
// Usage: cargo run --example marmoset_15d_lexicon_to_syntax --release [vocalizations_dir] [--resume]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::fs;
use std::io::Write;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::{MicroDynamicsExtractor, hdbscan::HdbscanClustering};

/// Marmoset call types (contexts)
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
enum CallType {
    Vocalization,  // General/unclassified vocalizations (largest category)
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

    fn description(&self) -> &'static str {
        match self {
            CallType::Vocalization => "General/unclassified vocalizations (largest category)",
            CallType::Phee => "Long-distance harmonic communication",
            CallType::Twitter => "Rapid high-pitched social calls",
            CallType::Trill => "Rapid frequency-modulated calls",
            CallType::Tsik => "Short sharp alarm calls",
            CallType::Seep => "Soft contact calls",
            CallType::Infant => "Infant distress calls",
            CallType::Unknown => "Uncharacterized vocalization",
        }
    }
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
                        // i24 is a tuple struct with i32.0 as inner value
                        // Normalize 24-bit signed int to [-1, 1]
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

/// Phrase candidate for lexicon discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseCandidate {
    phrase_id: String,
    file_name: String,
    call_type: CallType,
    features: Vec<f32>,  // 15D feature vector
    duration_ms: f32,
}

/// Vocabulary word (cluster of similar phrases)
#[derive(Debug, Clone)]
struct VocabWord {
    word_id: usize,
    representative_features: Vec<f32>,
    member_phrases: Vec<String>,
    contexts: HashSet<CallType>,  // Which call types use this word
}

/// Syntactic analysis results for a context
#[derive(Debug, Clone)]
struct ContextSyntaxResults {
    context: CallType,
    num_phrases: usize,
    vocabulary_size: usize,
    unique_words: usize,
    shared_words: usize,
    avg_sequence_length: f64,
    word_frequency_distribution: Vec<(usize, usize)>,  // (word_id, frequency)
}

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    completed_contexts: Vec<String>,  // Names of completed call types
    total_phrases: usize,
    timestamp_seconds: u64,
    phase2_completed: bool,  // Global vocabulary discovery completed
    phase3_completed: bool,  // Context-specific analysis completed
}

impl Default for CheckpointData {
    fn default() -> Self {
        Self {
            completed_contexts: Vec::new(),
            total_phrases: 0,
            timestamp_seconds: 0,
            phase2_completed: false,
            phase3_completed: false,
        }
    }
}

/// Checkpoint manager for saving/resuming progress
struct CheckpointManager {
    checkpoints_dir: PathBuf,
}

impl CheckpointManager {
    fn new(vocalizations_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let checkpoints_dir = vocalizations_dir.join(".checkpoints");
        fs::create_dir_all(&checkpoints_dir)?;
        Ok(Self { checkpoints_dir })
    }

    fn checkpoint_path(&self) -> PathBuf {
        self.checkpoints_dir.join("progress.json")
    }

    /// Get the phrases checkpoint path for a specific call type
    fn phrases_checkpoint_path(&self, call_type: &str) -> PathBuf {
        self.checkpoints_dir.join(format!("phrases_{}.json", call_type))
    }

    /// Load checkpoint if exists
    fn load_checkpoint(&self) -> Result<Option<CheckpointData>, Box<dyn std::error::Error>> {
        let path = self.checkpoint_path();
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)?;

        // Handle backward compatibility: try to load old format first
        // Old format had phase2_completed and phase3_completed missing
        let data: serde_json::Value = serde_json::from_str(&content)?;

        // Check if new fields exist, if not use defaults
        let phase2_completed = data.get("phase2_completed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let phase3_completed = data.get("phase3_completed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Parse the rest with defaults
        let completed_contexts: Vec<String> = data.get("completed_contexts")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let total_phrases = data.get("total_phrases")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let timestamp_seconds = data.get("timestamp_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        Ok(Some(CheckpointData {
            completed_contexts,
            total_phrases,
            timestamp_seconds,
            phase2_completed,
            phase3_completed,
        }))
    }

    /// Save checkpoint after completing a call type
    fn save_checkpoint(
        &self,
        completed_contexts: &[String],
        total_phrases: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Load existing checkpoint to preserve phase completion flags
        let mut data = self.load_checkpoint()?.unwrap_or_default();

        data.completed_contexts = completed_contexts.to_vec();
        data.total_phrases = total_phrases;
        data.timestamp_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let path = self.checkpoint_path();
        let json = serde_json::to_string_pretty(&data)?;
        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Mark Phase 2 as completed
    fn save_phase2_checkpoint(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut data = self.load_checkpoint()?.unwrap_or_default();
        data.phase2_completed = true;
        data.timestamp_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let path = self.checkpoint_path();
        let json = serde_json::to_string_pretty(&data)?;
        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Mark Phase 3 as completed
    fn save_phase3_checkpoint(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut data = self.load_checkpoint()?.unwrap_or_default();
        data.phase3_completed = true;
        data.timestamp_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let path = self.checkpoint_path();
        let json = serde_json::to_string_pretty(&data)?;
        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Get the Phase 2 checkpoint path (global vocabulary)
    fn phase2_checkpoint_path(&self) -> PathBuf {
        self.checkpoints_dir.join("phase2_global_vocabulary.json")
    }

    /// Get the Phase 3 checkpoint path (context results)
    fn phase3_checkpoint_path(&self) -> PathBuf {
        self.checkpoints_dir.join("phase3_context_results.json")
    }

    /// Save Phase 2 results (global vocabulary)
    fn save_phase2_results(
        &self,
        vocabulary: &[VocabWord],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.phase2_checkpoint_path();

        // Convert VocabWord to a serializable format
        let serializable: Vec<serde_json::Value> = vocabulary.iter().map(|word| {
            serde_json::json!({
                "word_id": word.word_id,
                "representative_features": word.representative_features,
                "member_phrases": word.member_phrases,
                "contexts": word.contexts.iter().map(|c| c.name()).collect::<Vec<_>>(),
            })
        }).collect();

        let json = serde_json::to_string_pretty(&serializable)?;
        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load Phase 2 results (global vocabulary)
    fn load_phase2_results(&self) -> Result<Option<Vec<VocabWord>>, Box<dyn std::error::Error>> {
        let path = self.phase2_checkpoint_path();
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)?;
        let data: Vec<serde_json::Value> = serde_json::from_str(&content)?;

        let vocabulary = data.into_iter().map(|value| {
            let word_id = value["word_id"].as_u64().unwrap() as usize;
            let representative_features: Vec<f32> = value["representative_features"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let member_phrases: Vec<String> = value["member_phrases"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let context_names: Vec<String> = value["contexts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();

            let contexts: HashSet<CallType> = context_names.iter()
                .filter_map(|name| match name.as_str() {
                    "Vocalization" => Some(CallType::Vocalization),
                    "Phee" => Some(CallType::Phee),
                    "Twitter" => Some(CallType::Twitter),
                    "Trill" => Some(CallType::Trill),
                    "Tsik" => Some(CallType::Tsik),
                    "Seep" => Some(CallType::Seep),
                    "Infant" => Some(CallType::Infant),
                    _ => None,
                })
                .collect();

            VocabWord {
                word_id,
                representative_features,
                member_phrases,
                contexts,
            }
        }).collect();

        Ok(Some(vocabulary))
    }

    /// Save Phase 3 results (context-specific analysis)
    fn save_phase3_results(
        &self,
        results: &[ContextSyntaxResults],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.phase3_checkpoint_path();

        let serializable: Vec<serde_json::Value> = results.iter().map(|r| {
            serde_json::json!({
                "context": r.context.name(),
                "num_phrases": r.num_phrases,
                "vocabulary_size": r.vocabulary_size,
                "unique_words": r.unique_words,
                "shared_words": r.shared_words,
                "avg_sequence_length": r.avg_sequence_length,
                "word_frequency_distribution": r.word_frequency_distribution,
            })
        }).collect();

        let json = serde_json::to_string_pretty(&serializable)?;
        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load Phase 3 results (context-specific analysis)
    fn load_phase3_results(&self) -> Result<Option<Vec<ContextSyntaxResults>>, Box<dyn std::error::Error>> {
        let path = self.phase3_checkpoint_path();
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)?;
        let data: Vec<serde_json::Value> = serde_json::from_str(&content)?;

        let results = data.into_iter().map(|value| {
            let context_name = value["context"].as_str().unwrap();
            let context = match context_name {
                "Vocalization" => CallType::Vocalization,
                "Phee" => CallType::Phee,
                "Twitter" => CallType::Twitter,
                "Trill" => CallType::Trill,
                "Tsik" => CallType::Tsik,
                "Seep" => CallType::Seep,
                "Infant" => CallType::Infant,
                _ => CallType::Unknown,
            };

            let word_freq_dist: Vec<(usize, usize)> = value["word_frequency_distribution"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                ))
                .collect();

            ContextSyntaxResults {
                context,
                num_phrases: value["num_phrases"].as_u64().unwrap() as usize,
                vocabulary_size: value["vocabulary_size"].as_u64().unwrap() as usize,
                unique_words: value["unique_words"].as_u64().unwrap() as usize,
                shared_words: value["shared_words"].as_u64().unwrap() as usize,
                avg_sequence_length: value["avg_sequence_length"].as_f64().unwrap(),
                word_frequency_distribution: word_freq_dist,
            }
        }).collect();

        Ok(Some(results))
    }

    /// Save phrases for a specific call type
    fn save_phrases_checkpoint(
        &self,
        call_type: &str,
        phrases: &[PhraseCandidate],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.phrases_checkpoint_path(call_type);
        let json = serde_json::to_string(phrases)?;
        let mut file = fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load phrases for a specific call type from checkpoint
    fn load_phrases_checkpoint(
        &self,
        call_type: &str,
    ) -> Result<Option<Vec<PhraseCandidate>>, Box<dyn std::error::Error>> {
        let path = self.phrases_checkpoint_path(call_type);
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)?;
        let phrases: Vec<PhraseCandidate> = serde_json::from_str(&content)?;
        Ok(Some(phrases))
    }
}

/// Check if a phrase has already been processed (exists in checkpoint)
fn is_phrase_processed(
    processed_phrase_ids: &HashSet<String>,
    phrase_id: &str,
) -> bool {
    processed_phrase_ids.contains(phrase_id)
}

/// Extract 15D features from audio
fn extract_15d_features(
    audio: &[f32],
    sample_rate: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features = extractor.extract_15d_marmoset(audio)?;
    Ok(features.to_array().to_vec())
}

/// HDBSCAN-based clustering for vocabulary discovery
///
/// Uses HDBSCAN to automatically discover the vocabulary structure
/// without requiring a fixed similarity threshold.
///
/// **Chunked Processing:** To avoid memory exhaustion on large datasets,
/// this function processes the data in chunks and hierarchically merges
/// the results.
///
/// **Memory Safety:** Chunk size can be configured via MARMOSET_CHUNK_SIZE
/// environment variable. Default is 3000 for WSL compatibility.
fn discover_vocabulary(
    phrases: &[PhraseCandidate],
    min_cluster_size: usize,
    min_samples: usize,
    hdbscan_config: &HdbscanConfig,
) -> Vec<VocabWord> {
    if phrases.is_empty() {
        return Vec::new();
    }

    let n_samples = phrases.len();
    let n_features = phrases[0].features.len();

    // Skip clustering mode: treat each phrase as its own word
    if hdbscan_config.skip_clustering {
        println!("  ⏭ Skip-clustering mode: treating each of {} phrases as its own word", n_samples);

        return phrases.iter().enumerate().map(|(i, phrase)| {
            let mut contexts = HashSet::new();
            contexts.insert(phrase.call_type);

            VocabWord {
                word_id: i,
                representative_features: phrase.features.clone(),
                member_phrases: vec![phrase.phrase_id.clone()],
                contexts,
            }
        }).collect();
    }

    println!("  📊 Running HDBSCAN clustering on {} phrases ({}D features)...", n_samples, n_features);
    println!("     ├─ min_cluster_size: {}", min_cluster_size);
    println!("     └─ min_samples: {}", min_samples);
    if hdbscan_config.no_merge {
        println!("     ⚠ No-merge mode: will skip centroid merging");
    }

    // CHUNK_SIZE: Process in chunks of this many phrases to avoid memory exhaustion
    // For WSL with limited RAM, smaller chunk sizes are critical
    // Can be overridden via MARMOSET_CHUNK_SIZE environment variable
    let chunk_size: usize = std::env::var("MARMOSET_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000); // Default to 3000 for WSL safety (was 8000, too large)

    // If dataset is small enough, use single-pass clustering
    if n_samples <= chunk_size {
        return discover_vocabulary_single_pass(phrases, min_cluster_size, min_samples);
    }

    // LARGE DATASET: Use chunked processing with hierarchical merging
    println!("  📦 Large dataset detected, using chunked processing (chunk_size = {})...", chunk_size);

    // Phase 1: Cluster each chunk independently
    let mut chunk_vocabularies: Vec<Vec<VocabWord>> = Vec::new();
    let mut total_chunks = (n_samples + chunk_size - 1) / chunk_size;

    for (chunk_idx, chunk) in phrases.chunks(chunk_size).enumerate() {
        println!("  🔄 Processing chunk {}/{} ({} phrases)...",
                 chunk_idx + 1, total_chunks, chunk.len());

        // Adaptive parameters for chunk-level clustering
        // Smaller chunks need smaller min_cluster_size
        let chunk_min_cluster_size = ((chunk.len() as f64).sqrt() as usize).max(5);
        let chunk_min_samples = (chunk_min_cluster_size * 3) / 4;

        // Process chunk and collect results
        let chunk_vocabulary = discover_vocabulary_single_pass(
            chunk,
            chunk_min_cluster_size,
            chunk_min_samples,
        );

        println!("     → Found {} words in chunk {}", chunk_vocabulary.len(), chunk_idx + 1);

        // Store results and explicitly drop chunk data to free memory
        chunk_vocabularies.push(chunk_vocabulary);

        // Explicit memory cleanup hint for chunks
        if chunk_idx % 5 == 0 && chunk_idx > 0 {
            // Every 5 chunks, hint at garbage collection
            // This helps on systems with limited memory
            println!("     💾 Memory checkpoint at chunk {}...", chunk_idx);
        }
    }

    // Phase 2: Merge vocabularies by clustering their centroids
    // (Skip if --no-merge flag is set)
    if hdbscan_config.no_merge {
        println!();
        println!("  📋 No-merge mode: returning {} chunk-level vocabularies without merging...",
                 chunk_vocabularies.len());

        // Flatten all chunk vocabularies into a single list
        let mut flat_vocabulary: Vec<VocabWord> = Vec::new();
        let mut word_id = 0;
        for vocab in chunk_vocabularies {
            for mut word in vocab {
                word.word_id = word_id;
                word_id += 1;
                flat_vocabulary.push(word);
            }
        }
        println!("     → Total words: {}", flat_vocabulary.len());
        return flat_vocabulary;
    }

    // Phase 2: Merge vocabularies by clustering their centroids
    println!();
    println!("  🔄 Phase 2: Merging {} chunk vocabularies...", chunk_vocabularies.len());

    // Collect all centroids and track their source members
    let mut all_centroids: Vec<Vec<f32>> = Vec::new();
    let mut all_members: Vec<Vec<String>> = Vec::new();
    let mut all_contexts: Vec<HashSet<CallType>> = Vec::new();

    for vocab in &chunk_vocabularies {
        for word in vocab {
            all_centroids.push(word.representative_features.clone());
            all_members.push(word.member_phrases.clone());
            all_contexts.push(word.contexts.clone());
        }
    }

    if all_centroids.is_empty() {
        return Vec::new();
    }

    // Cluster the centroids using adaptive parameters
    let n_centroids = all_centroids.len();
    let merge_min_cluster_size = ((n_centroids as f64).ln() as usize).max(3);
    let merge_min_samples = (merge_min_cluster_size * 3) / 4;

    println!("     ├─ Clustering {} centroids", n_centroids);
    println!("     ├─ merge_min_cluster_size: {}", merge_min_cluster_size);
    println!("     └─ merge_min_samples: {}", merge_min_samples);

    // Build feature matrix for centroids
    let mut centroid_matrix = ndarray::Array2::zeros((n_centroids, n_features));
    for (i, centroid) in all_centroids.iter().enumerate() {
        for (j, &val) in centroid.iter().enumerate() {
            centroid_matrix[[i, j]] = val as f64;
        }
    }

    // Run HDBSCAN on centroids
    let hdbscan = match HdbscanClustering::new(merge_min_cluster_size, merge_min_samples) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  ⚠ Failed to create HDBSCAN for merging: {:?}, using all centroids as words", e);
            return create_vocabulary_from_centroids(all_centroids, all_members, all_contexts);
        }
    };

    let labels = match hdbscan.fit_predict(&centroid_matrix) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  ⚠ HDBSCAN merge failed: {:?}, using all centroids as words", e);
            return create_vocabulary_from_centroids(all_centroids, all_members, all_contexts);
        }
    };

    // Group centroids by merged cluster
    let mut merged_clusters: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut noise_count = 0;

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            merged_clusters.entry(label).or_insert_with(Vec::new).push(i);
        } else {
            noise_count += 1;
        }
    }

    println!("  ✅ Merged into {} final vocabulary words ({} noise centroids excluded)",
             merged_clusters.len(), noise_count);

    // Build final vocabulary from merged clusters
    let mut final_vocabulary: Vec<VocabWord> = Vec::new();
    for (merged_id, centroid_indices) in merged_clusters {
        // Merge all members and contexts from this cluster
        let mut merged_members: Vec<String> = Vec::new();
        let mut merged_contexts: HashSet<CallType> = HashSet::new();
        let mut merged_centroid = vec![0.0f32; n_features];

        for &idx in &centroid_indices {
            merged_members.extend(all_members[idx].iter().cloned());
            merged_contexts = merged_contexts.union(&all_contexts[idx]).cloned().collect();
            for (j, &val) in all_centroids[idx].iter().enumerate() {
                merged_centroid[j] += val;
            }
        }

        // Average the centroid
        for val in merged_centroid.iter_mut() {
            *val /= centroid_indices.len() as f32;
        }

        final_vocabulary.push(VocabWord {
            word_id: merged_id as usize,
            representative_features: merged_centroid,
            member_phrases: merged_members,
            contexts: merged_contexts,
        });
    }

    final_vocabulary.sort_by_key(|w| w.word_id);
    final_vocabulary
}

/// Single-pass HDBSCAN clustering for smaller datasets
///
/// This function is memory-safe and will fall back to smaller sub-chunks
/// if the input is still too large for a single HDBSCAN call.
fn discover_vocabulary_single_pass(
    phrases: &[PhraseCandidate],
    min_cluster_size: usize,
    min_samples: usize,
) -> Vec<VocabWord> {
    if phrases.is_empty() {
        return Vec::new();
    }

    let n_samples = phrases.len();
    let n_features = phrases[0].features.len();

    // Memory safety check: if dataset is still too large for single HDBSCAN call,
    // recursively split into smaller chunks
    // HDBSCAN builds O(n²) distance matrices, so we need to be conservative
    const SAFE_HDBSCAN_MAX_SAMPLES: usize = 2000;

    if n_samples > SAFE_HDBSCAN_MAX_SAMPLES {
        // Recursively process in smaller sub-chunks
        let mut results: Vec<VocabWord> = Vec::new();
        for sub_chunk in phrases.chunks(SAFE_HDBSCAN_MAX_SAMPLES) {
            let sub_results = discover_vocabulary_single_pass(
                sub_chunk,
                min_cluster_size,
                min_samples,
            );
            results.extend(sub_results);
        }
        return results;
    }

    // Build feature matrix for HDBSCAN (convert f32 to f64)
    let mut feature_matrix = ndarray::Array2::zeros((n_samples, n_features));
    for (i, phrase) in phrases.iter().enumerate() {
        for (j, &val) in phrase.features.iter().enumerate() {
            feature_matrix[[i, j]] = val as f64;
        }
    }

    // Run HDBSCAN clustering
    let hdbscan = match HdbscanClustering::new(min_cluster_size, min_samples) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  ⚠ Failed to create HDBSCAN: {:?}, falling back to single cluster", e);
            return vec![create_single_word_cluster(phrases, 0)];
        }
    };

    let labels = match hdbscan.fit_predict(&feature_matrix) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  ⚠ HDBSCAN failed: {:?}, falling back to single cluster", e);
            return vec![create_single_word_cluster(phrases, 0)];
        }
    };

    // Group phrases by cluster label (ignore noise points with label -1)
    let mut cluster_map: HashMap<i32, Vec<&PhraseCandidate>> = HashMap::new();
    let mut noise_count = 0;

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            cluster_map.entry(label).or_insert_with(Vec::new).push(&phrases[i]);
        } else {
            noise_count += 1;
        }
    }

    println!("     → Found {} clusters ({} noise points excluded)", cluster_map.len(), noise_count);

    // Convert clusters to VocabWord structs
    cluster_map
        .into_iter()
        .map(|(cluster_id, cluster)| {
            let word_id = cluster_id as usize;

            // Compute centroid as representative features
            let n_features = cluster[0].features.len();
            let mut centroid = vec![0.0f32; n_features];
            for phrase in &cluster {
                for (i, &val) in phrase.features.iter().enumerate() {
                    centroid[i] += val;
                }
            }
            for val in centroid.iter_mut() {
                *val /= cluster.len() as f32;
            }

            let member_phrases: Vec<String> = cluster.iter().map(|p| p.phrase_id.clone()).collect();
            let contexts: HashSet<CallType> = cluster.iter().map(|p| p.call_type).collect();

            VocabWord {
                word_id,
                representative_features: centroid,
                member_phrases,
                contexts,
            }
        })
        .collect()
}

/// Create vocabulary directly from centroids (fallback when merging fails)
fn create_vocabulary_from_centroids(
    centroids: Vec<Vec<f32>>,
    members: Vec<Vec<String>>,
    contexts: Vec<HashSet<CallType>>,
) -> Vec<VocabWord> {
    centroids.into_iter().zip(members).zip(contexts)
        .enumerate()
        .map(|(word_id, ((centroid, member_phrases), contexts))| {
            VocabWord {
                word_id,
                representative_features: centroid,
                member_phrases,
                contexts,
            }
        })
        .collect()
}

/// Create a single word cluster as fallback
fn create_single_word_cluster(phrases: &[PhraseCandidate], word_id: usize) -> VocabWord {
    let n_features = phrases[0].features.len();
    let mut centroid = vec![0.0f32; n_features];
    for phrase in phrases {
        for (i, &val) in phrase.features.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for val in centroid.iter_mut() {
        *val /= phrases.len() as f32;
    }

    VocabWord {
        word_id,
        representative_features: centroid,
        member_phrases: phrases.iter().map(|p| p.phrase_id.clone()).collect(),
        contexts: phrases.iter().map(|p| p.call_type).collect(),
    }
}

/// Analyze a single context (call type) using HDBSCAN clustering
fn analyze_context(
    phrases: &[PhraseCandidate],
    call_type: CallType,
    min_cluster_size: usize,
    min_samples: usize,
    hdbscan_config: &HdbscanConfig,
) -> Result<ContextSyntaxResults, Box<dyn std::error::Error>> {
    if phrases.is_empty() {
        return Ok(ContextSyntaxResults {
            context: call_type,
            num_phrases: 0,
            vocabulary_size: 0,
            unique_words: 0,
            shared_words: 0,
            avg_sequence_length: 0.0,
            word_frequency_distribution: Vec::new(),
        });
    }

    // Discover vocabulary for this context using HDBSCAN
    let vocabulary = discover_vocabulary(phrases, min_cluster_size, min_samples, hdbscan_config);

    // Count unique vs shared words
    let unique_words = vocabulary.iter()
        .filter(|w| w.contexts.len() == 1 && w.contexts.contains(&call_type))
        .count();

    let shared_words = vocabulary.iter()
        .filter(|w| w.contexts.len() > 1)
        .count();

    // Build word frequency distribution
    let mut word_counts: HashMap<usize, usize> = HashMap::new();
    for phrase in phrases {
        // Find which word this phrase belongs to
        for (word_idx, word) in vocabulary.iter().enumerate() {
            if word.member_phrases.contains(&phrase.phrase_id) {
                *word_counts.entry(word_idx).or_insert(0) += 1;
                break;
            }
        }
    }

    let mut word_freq_dist: Vec<_> = word_counts.into_iter().collect();
    word_freq_dist.sort_by(|a, b| b.1.cmp(&a.1));  // Sort by frequency descending

    Ok(ContextSyntaxResults {
        context: call_type,
        num_phrases: phrases.len(),
        vocabulary_size: vocabulary.len(),
        unique_words,
        shared_words,
        avg_sequence_length: 1.0,  // Each phrase is one "word" in this analysis
        word_frequency_distribution: word_freq_dist,
    })
}

/// Scan vocalizations directory and group files by call type
fn scan_vocalizations_dir(
    vocalizations_dir: &Path,
) -> Result<HashMap<CallType, Vec<String>>, Box<dyn std::error::Error>> {
    let mut context_files: HashMap<CallType, Vec<String>> = HashMap::new();

    let entries = std::fs::read_dir(vocalizations_dir)?;
    for entry in entries {
        let entry = entry?;
        let dir_path = entry.path();

        if !dir_path.is_dir() {
            continue;
        }

        // Check each subdirectory for FLAC files
        let file_entries = std::fs::read_dir(&dir_path)?;
        for file_entry in file_entries {
            let file_entry = file_entry?;
            let file_path = file_entry.path();

            if !file_path.is_file() {
                continue;
            }

            let filename = file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if !filename.to_lowercase().ends_with(".flac") {
                continue;
            }

            let call_type = CallType::from_filename(filename);
            if call_type != CallType::Unknown {
                let full_path = file_path.to_str().ok_or("Invalid path")?.to_string();
                context_files.entry(call_type).or_insert_with(Vec::new).push(full_path);
            }
        }
    }

    Ok(context_files)
}

/// Run Phase 2 clustering and save results
fn run_phase2_clustering(
    all_phrases: &[PhraseCandidate],
    checkpoint_manager: &CheckpointManager,
    hdbscan_config: &HdbscanConfig,
) -> Result<Vec<VocabWord>, Box<dyn std::error::Error>> {
    // HDBSCAN parameters for global vocabulary discovery
    let n_total = all_phrases.len();
    let (min_cluster_size, min_samples) = hdbscan_config.get_global_params(n_total);

    println!("Global HDBSCAN parameters:");
    if hdbscan_config.min_cluster_size.is_some() {
        println!("  min_cluster_size: {} (custom)", min_cluster_size);
    } else {
        println!("  min_cluster_size: {} (based on sqrt(n))", min_cluster_size);
    }
    if hdbscan_config.min_samples.is_some() {
        println!("  min_samples: {} (custom)", min_samples);
    } else {
        println!("  min_samples: {} (75% of min_cluster_size)", min_samples);
    }
    println!();

    let global_vocabulary = discover_vocabulary(all_phrases, min_cluster_size, min_samples, hdbscan_config);

    // Save Phase 2 results
    checkpoint_manager.save_phase2_results(&global_vocabulary)?;
    checkpoint_manager.save_phase2_checkpoint()?;

    Ok(global_vocabulary)
}

/// Run Phase 3 analysis and return results
fn run_phase3_analysis(
    context_phrases: &Mutex<HashMap<CallType, Vec<PhraseCandidate>>>,
    hdbscan_config: &HdbscanConfig,
) -> Result<Vec<ContextSyntaxResults>, Box<dyn std::error::Error>> {
    let mut context_results: Vec<ContextSyntaxResults> = Vec::new();

    for call_type in [
        CallType::Vocalization,
        CallType::Phee,
        CallType::Twitter,
        CallType::Trill,
        CallType::Tsik,
        CallType::Seep,
        CallType::Infant,
    ] {
        let phrases_to_analyze = {
            let context_phrases_guard = context_phrases.lock().unwrap();
            context_phrases_guard.get(&call_type).cloned()
        };

        if let Some(phrases) = phrases_to_analyze {
            // Adaptive HDBSCAN parameters per context
            let n_phrases = phrases.len();
            let (context_min_cluster_size, context_min_samples) = hdbscan_config.get_context_params(n_phrases);

            match analyze_context(&phrases, call_type, context_min_cluster_size, context_min_samples, hdbscan_config) {
                Ok(results) => {
                    context_results.push(results);
                }
                Err(e) => {
                    println!("  ❌ Analysis failed for {}: {}", call_type.name(), e);
                }
            }
        }
    }

    Ok(context_results)
}

/// HDBSCAN parameter configuration
#[derive(Debug, Clone)]
struct HdbscanConfig {
    min_cluster_size: Option<usize>,
    min_samples: Option<usize>,
    no_merge: bool,  // Skip hierarchical merging for chunk-level granularity
    skip_clustering: bool,  // Skip HDBSCAN entirely - treat each phrase as its own word
}

impl HdbscanConfig {
    fn get_global_params(&self, n_samples: usize) -> (usize, usize) {
        let min_cluster_size = self.min_cluster_size.unwrap_or_else(|| {
            (n_samples as f64).sqrt() as usize
        });
        let min_samples = self.min_samples.unwrap_or_else(|| {
            (min_cluster_size * 3) / 4
        });
        (min_cluster_size, min_samples)
    }

    fn get_context_params(&self, n_samples: usize) -> (usize, usize) {
        let min_cluster_size = self.min_cluster_size.unwrap_or_else(|| {
            ((n_samples as f64).ln() as usize).max(5)
        });
        let min_samples = self.min_samples.unwrap_or_else(|| {
            (min_cluster_size * 3) / 4
        });
        (min_cluster_size, min_samples)
    }

    fn is_custom(&self) -> bool {
        self.min_cluster_size.is_some() || self.min_samples.is_some() || self.no_merge || self.skip_clustering
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Marmoset Lexicon-to-Syntax Analysis (15D RFE-Optimized)                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut resume_from_checkpoint = false;
    let mut recluster = false; // Force re-run phases 2/3 even if checkpointed
    let mut no_merge = false; // Skip centroid merging for more granular results
    let mut skip_clustering = false; // Skip HDBSCAN clustering entirely
    let mut vocalizations_dir = "/home/sheel/birdsong_analysis/data/Vocalizations".to_string();

    // Check for help flag
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: cargo run --example marmoset_15d_lexicon_to_syntax --release [OPTIONS] [vocalizations_dir]");
        println!();
        println!("Options:");
        println!("  --resume                Resume from checkpoint (skip Phase 1)");
        println!("  --recluster             Force re-run Phase 2 & 3 even if checkpointed");
        println!("  --no-merge              Skip centroid merging for more granular chunk-level results");
        println!("  --skip-clustering       Skip HDBSCAN entirely - treat each phrase as its own word");
        println!("  --min-cluster-size N    Override min_cluster_size for HDBSCAN (default: sqrt(n))");
        println!("  --min-samples N         Override min_samples for HDBSCAN (default: 75% of min_cluster_size)");
        println!("  --help, -h              Show this help message");
        println!();
        println!("Environment Variables:");
        println!("  MARMOSET_CHUNK_SIZE     Override chunk size for memory-safe processing (default: 3000)");
        println!();
        println!("Examples:");
        println!("  # Resume with default parameters");
        println!("  cargo run --example marmoset_15d_lexicon_to_syntax --release -- --resume");
        println!();
        println!("  # Tune for more granular vocabulary (smaller clusters)");
        println!("  cargo run --example marmoset_15d_lexicon_to_syntax --release -- --resume --min-cluster-size 50");
        println!();
        println!("  # Skip merging to preserve chunk-level granularity");
        println!("  cargo run --example marmoset_15d_lexicon_to_syntax --release -- --resume --no-merge");
        println!();
        println!("  # Skip HDBSCAN clustering entirely - each phrase is a word");
        println!("  cargo run --example marmoset_15d_lexicon_to_syntax --release -- --resume --skip-clustering");
        println!();
        println!("  # Force recluster with custom parameters");
        println!("  cargo run --example marmoset_15d_lexicon_to_syntax --release -- --resume --recluster --min-cluster-size 100");
        return Ok(());
    }

    // Parse arguments in a single pass to handle all flags correctly
    let mut i = 1;
    let mut min_cluster_size = None;
    let mut min_samples = None;

    while i < args.len() {
        match args[i].as_str() {
            "--resume" => {
                resume_from_checkpoint = true;
            }
            "--recluster" => {
                recluster = true;
                println!("🔄 Force recluster flag: will re-run phases 2/3 even if checkpointed");
            }
            "--no-merge" => {
                no_merge = true;
                println!("📋 No-merge mode: skipping centroid merging for chunk-level granularity");
            }
            "--skip-clustering" => {
                skip_clustering = true;
                println!("⏭ Skip-clustering mode: treating each phrase as its own word");
            }
            "--min-cluster-size" => {
                if i + 1 < args.len() {
                    if let Ok(size) = args[i + 1].parse::<usize>() {
                        min_cluster_size = Some(size);
                        println!("📊 Using custom min_cluster_size: {}", size);
                    }
                    i += 1; // Skip the next argument
                }
            }
            "--min-samples" => {
                if i + 1 < args.len() {
                    if let Ok(samples) = args[i + 1].parse::<usize>() {
                        min_samples = Some(samples);
                        println!("📊 Using custom min_samples: {}", samples);
                    }
                    i += 1; // Skip the next argument
                }
            }
            arg if i == args.len() - 1 && !arg.starts_with("--") => {
                // Last argument that doesn't start with -- is the directory
                vocalizations_dir = arg.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    let hdbscan_config = HdbscanConfig { min_cluster_size, min_samples, no_merge, skip_clustering };

    let vocalizations_dir = Path::new(&vocalizations_dir);

    if !vocalizations_dir.exists() {
        println!("❌ Vocalizations directory not found: {}", vocalizations_dir.display());
        println!("   Usage: cargo run --example marmoset_15d_lexicon_to_syntax --release <vocalizations_dir> [--resume]");
        return Err("Directory not found".into());
    }

    // Initialize checkpoint manager
    let checkpoint_manager = CheckpointManager::new(vocalizations_dir)?;

    // Load checkpoint if resuming
    let mut completed_contexts: HashSet<String> = HashSet::new();
    let mut loaded_phrases: Vec<PhraseCandidate> = Vec::new();
    let mut phase2_completed = false;
    let mut phase3_completed = false;

    if resume_from_checkpoint {
        println!("🔄 Resuming from checkpoint...");
        if let Some(checkpoint) = checkpoint_manager.load_checkpoint()? {
            println!("   Checkpoint found:");
            println!("     - Completed contexts: {:?}", checkpoint.completed_contexts);
            println!("     - Total phrases: {}", checkpoint.total_phrases);
            println!("     - Phase 2 completed: {}", checkpoint.phase2_completed);
            println!("     - Phase 3 completed: {}", checkpoint.phase3_completed);

            completed_contexts = checkpoint.completed_contexts.into_iter().collect();

            // If --recluster flag is set, force re-run phases 2/3
            if recluster || hdbscan_config.is_custom() {
                println!("   🔄 Recluster requested or custom HDBSCAN params detected:");
                if recluster {
                    println!("     - --recluster flag set");
                }
                if hdbscan_config.is_custom() {
                    println!("     - Custom HDBSCAN parameters: {:?}", hdbscan_config);
                }
                println!("     - Will re-run Phase 2 & 3 with new parameters");
                phase2_completed = false;
                phase3_completed = false;
            } else {
                phase2_completed = checkpoint.phase2_completed;
                phase3_completed = checkpoint.phase3_completed;
            }

            // Load all saved phrases
            for context_name in &completed_contexts {
                if let Some(phrases) = checkpoint_manager.load_phrases_checkpoint(context_name)? {
                    println!("   Loaded {} phrases from {}", phrases.len(), context_name);
                    loaded_phrases.extend(phrases);
                }
            }
            println!("   Total loaded phrases: {}", loaded_phrases.len());
            println!();
        } else {
            println!("   No checkpoint found, starting fresh...");
            println!();
        }
    }

    println!("📂 Scanning vocalizations directory: {}", vocalizations_dir.display());
    println!();

    // Scan for marmoset vocalization files
    let context_files = scan_vocalizations_dir(vocalizations_dir)?;

    let total_files: usize = context_files.values().map(|v| v.len()).sum();
    println!("✅ Discovered {} FLAC files across {} call types", total_files, context_files.len());
    println!();

    // Display file counts per call type
    println!("Call Type Distribution:");
    println!("======================");
    for call_type in [
        CallType::Vocalization,
        CallType::Phee,
        CallType::Twitter,
        CallType::Trill,
        CallType::Tsik,
        CallType::Seep,
        CallType::Infant,
    ] {
        let count = context_files.get(&call_type).map(|v| v.len()).unwrap_or(0);
        let status = if completed_contexts.contains(&call_type.name().to_string()) {
            "✓"
        } else {
            ""
        };
        if count > 0 {
            println!("  [{:1}] {:15} {:>8} files | {}", status, call_type.name(), count, call_type.description());
        }
    }
    println!();

    // Extract features from all files
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 1: Feature Extraction (15D RFE-Optimized)                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let sample_rate = 96000;

    // If resuming, load existing phrases into memory
    let loaded_phrases_clone = loaded_phrases.clone();
    let context_phrases: Mutex<HashMap<CallType, Vec<PhraseCandidate>>> = Mutex::new(HashMap::new());

    // Load checkpoint phrases into context map
    for phrase in &loaded_phrases_clone {
        let mut ctx = context_phrases.lock().unwrap();
        ctx.entry(phrase.call_type).or_insert_with(Vec::new).push(phrase.clone());
    }

    // Process all call types in parallel
    // Each call type returns its phrases for later merging
    let call_types: Vec<_> = context_files.iter().collect();
    let total_call_types = call_types.len();

    // Clone the vocalizations dir path for use in parallel
    let vocalizations_dir_path = vocalizations_dir.to_path_buf();

    println!("🚀 Processing {} call types in parallel...", total_call_types);
    println!();

    // Process call types in parallel, collecting results
    let call_type_results: Vec<_> = call_types
        .par_iter()
        .map(|(call_type, files)| {
            let call_type_name = call_type.name().to_string();

            // Skip if already completed
            if completed_contexts.contains(&call_type_name) {
                println!("[{}] ⏭ Skipping {} (already completed)",
                         std::thread::current().name().unwrap_or("unknown"), call_type_name);
                return None;
            }

            println!("[{}] 🔄 Processing {} files for {} in parallel...",
                     std::thread::current().name().unwrap_or("unknown"),
                     files.len(),
                     call_type_name);

            let start_time = std::time::Instant::now();
            let counter = Mutex::new(0usize);
            let total_files = files.len();
            let phrases_for_call_type: Mutex<Vec<PhraseCandidate>> = Mutex::new(Vec::new());

            files.par_iter().for_each(|file_path| {
                let path = Path::new(file_path);
                let filename = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                match load_flac_file(path) {
                    Ok(audio) => {
                        // Extract 15D features
                        match extract_15d_features(&audio, sample_rate) {
                            Ok(features) => {
                                let duration_ms = audio.len() as f32 / sample_rate as f32 * 1000.0;

                                let phrase = PhraseCandidate {
                                    phrase_id: filename.to_string(),
                                    file_name: filename.to_string(),
                                    call_type: **call_type,
                                    features,
                                    duration_ms,
                                };

                                // Thread-safe insertion
                                {
                                    let mut phrases = phrases_for_call_type.lock().unwrap();
                                    phrases.push(phrase);
                                }

                                // Update progress counter
                                {
                                    let mut count = counter.lock().unwrap();
                                    *count += 1;
                                    if *count % 10000 == 0 || *count == total_files {
                                        let elapsed = start_time.elapsed().as_secs_f64();
                                        let rate = *count as f64 / elapsed;
                                        let remaining = (total_files - *count) as f64 / rate;
                                        println!("[{}] Progress: {}/{} ({:.1}%) | {:.1} files/sec | ETA: {:.1}s",
                                                 std::thread::current().name().unwrap_or("unknown"),
                                                 *count, total_files,
                                                 *count as f64 / total_files as f64 * 100.0,
                                                 rate,
                                                 remaining);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[{}] Warning: Feature extraction failed for {}: {}",
                                          std::thread::current().name().unwrap_or("unknown"), filename, e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[{}] Warning: Failed to load {}: {}",
                                  std::thread::current().name().unwrap_or("unknown"), filename, e);
                    }
                }
            });

            let elapsed = start_time.elapsed().as_secs_f64();
            println!("[{}] ✅ Completed {} in {:.1}s ({:.1} files/sec)",
                     std::thread::current().name().unwrap_or("unknown"),
                     call_type_name, elapsed, total_files as f64 / elapsed);

            // Collect phrases for this call type
            let phrases_for_type = phrases_for_call_type.into_inner().unwrap_or_default();

            // Create a local checkpoint manager for this call type
            let local_checkpoint_manager = CheckpointManager::new(&vocalizations_dir_path)
                .expect("Failed to create checkpoint manager");

            // Save checkpoint for this call type
            println!("[{}] Saving checkpoint...", std::thread::current().name().unwrap_or("unknown"));
            if let Err(e) = local_checkpoint_manager.save_phrases_checkpoint(&call_type_name, &phrases_for_type) {
                eprintln!("[{}] Warning: Failed to save phrases checkpoint: {}",
                          std::thread::current().name().unwrap_or("unknown"), e);
            }

            Some((*call_type, call_type_name, phrases_for_type))
        })
        .collect();

    // Merge results from all call types
    println!();
    println!("🔄 Merging results from all call types...");

    let mut merged_completed_contexts: Vec<String> = completed_contexts.iter().cloned().collect();
    let mut merged_context_phrases: HashMap<CallType, Vec<PhraseCandidate>> = HashMap::new();

    for result in call_type_results {
        if let Some((call_type, call_type_name, phrases)) = result {
            println!("  {} : {} phrases extracted", call_type_name, phrases.len());
            merged_context_phrases.insert(*call_type, phrases);
            merged_completed_contexts.push(call_type_name);
        }
    }

    // Merge all phrases from all call types
    let mut all_phrases: Vec<PhraseCandidate> = loaded_phrases;
    all_phrases.extend(merged_context_phrases
        .values()
        .flat_map(|v| v.iter().cloned())
        .collect::<Vec<_>>());

    // Update context_phrases with merged results
    *context_phrases.lock().unwrap() = merged_context_phrases;

    // Save final checkpoint
    if let Err(e) = checkpoint_manager.save_checkpoint(
        &merged_completed_contexts,
        all_phrases.len(),
    ) {
        eprintln!("    Warning: Failed to save progress checkpoint: {}", e);
    }

    println!();
    println!("✅ Feature extraction complete!");
    println!("   Total phrases extracted: {}", all_phrases.len());
    println!();

    // Discover global vocabulary (across all contexts) using HDBSCAN
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 2: Global Vocabulary Discovery (HDBSCAN)                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let global_vocabulary = if phase2_completed {
        println!("⏭ Phase 2 already completed, loading from checkpoint...");
        match checkpoint_manager.load_phase2_results()? {
            Some(vocab) => {
                println!("✅ Loaded {} words from checkpoint", vocab.len());
                println!();
                vocab
            }
            None => {
                println!("⚠ Checkpoint data not found, re-running Phase 2...");
                println!();
                run_phase2_clustering(&all_phrases, &checkpoint_manager, &hdbscan_config)?
            }
        }
    } else {
        run_phase2_clustering(&all_phrases, &checkpoint_manager, &hdbscan_config)?
    };

    println!();
    println!("Global Vocabulary Statistics:");
    println!("  Total vocabulary size: {} words", global_vocabulary.len());
    println!("  Clustering method: HDBSCAN");
    println!();

    // Analyze word sharing across contexts
    let mut universal_words = 0;
    let mut context_specific_words = 0;
    let mut shared_2_3_contexts = 0;

    for word in &global_vocabulary {
        match word.contexts.len() {
            1 => context_specific_words += 1,
            2..=3 => shared_2_3_contexts += 1,
            _ => universal_words += 1,
        }
    }

    println!("Word Sharing Across Contexts:");
    println!("  Universal (all 7 contexts):     {} words ({:.1}%)",
             universal_words,
             universal_words as f64 / global_vocabulary.len() as f64 * 100.0);
    println!("  Shared (2-3 contexts):          {} words ({:.1}%)",
             shared_2_3_contexts,
             shared_2_3_contexts as f64 / global_vocabulary.len() as f64 * 100.0);
    println!("  Context-specific (1 context):  {} words ({:.1}%)",
             context_specific_words,
             context_specific_words as f64 / global_vocabulary.len() as f64 * 100.0);
    println!();

    // Analyze each context separately
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 3: Cross-Context Syntax Analysis                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let context_results = if phase3_completed {
        println!("⏭ Phase 3 already completed, loading from checkpoint...");
        match checkpoint_manager.load_phase3_results()? {
            Some(results) => {
                println!("✅ Loaded {} context results from checkpoint", results.len());
                println!();
                results
            }
            None => {
                println!("⚠ Checkpoint data not found, re-running Phase 3...");
                println!();
                run_phase3_analysis(&context_phrases, &hdbscan_config)?
            }
        }
    } else {
        run_phase3_analysis(&context_phrases, &hdbscan_config)?
    };

    // Save Phase 3 results
    if !phase3_completed {
        checkpoint_manager.save_phase3_results(&context_results)?;
        checkpoint_manager.save_phase3_checkpoint()?;
    }

    // Display results for each context
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CONTEXT-SPECIFIC RESULTS                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    for results in &context_results {
        println!("┌─────────────────────────────────────────────────────────────────────────┐");
        println!("│ Context: {} ({})                                            │",
                 results.context.name(),
                 results.context.description());
        println!("└─────────────────────────────────────────────────────────────────────────┘");
        println!("  Total phrases:           {}", results.num_phrases);
        println!("  Vocabulary size:         {} words", results.vocabulary_size);
        println!("  Unique words:            {} (only in this context)", results.unique_words);
        println!("  Shared words:            {} (found in ≥2 contexts)", results.shared_words);
        println!();

        if !results.word_frequency_distribution.is_empty() {
            println!("  Top 10 Most Frequent Words:");
            for (i, (word_id, freq)) in results.word_frequency_distribution.iter().take(10).enumerate() {
                println!("    {:2}. Word {:>4} occurs {} time(s)", i + 1, word_id, freq);
            }
        }
        println!();
    }

    // Cross-context comparison
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CROSS-CONTEXT COMPARISON                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("{:<15} {:>10} {:>10} {:>12} {:>12} {:>12}",
             "Context", "Phrases", "Vocab", "Unique", "Shared", "Uniq%");
    println!("{}", "-".repeat(75));

    for results in &context_results {
        let uniq_pct = if results.vocabulary_size > 0 {
            results.unique_words as f64 / results.vocabulary_size as f64 * 100.0
        } else {
            0.0
        };

        println!("{:<15} {:>10} {:>10} {:>12} {:>12} {:>11.1}%",
                 results.context.name(),
                 results.num_phrases,
                 results.vocabulary_size,
                 results.unique_words,
                 results.shared_words,
                 uniq_pct);
    }
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         RESEARCH IMPLICATIONS                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    if universal_words > 0 {
        println!("✓ UNIVERSAL VOCABULARY DETECTED");
        println!("  → {} words shared across ALL call types", universal_words);
        println!("  → Suggests: Common acoustic building blocks in marmoset vocal production");
    } else {
        println!("~ NO UNIVERSAL VOCABULARY");
        println!("  → Each call type has distinct vocabulary");
        println!("  → Suggests: Call type-specific vocal production mechanisms");
    }
    println!();

    if context_specific_words > global_vocabulary.len() / 2 {
        println!("✓ HIGH CONTEXT SPECIFICITY");
        println!("  → {:.0}% of words are context-specific",
                 context_specific_words as f64 / global_vocabulary.len() as f64 * 100.0);
        println!("  → Suggests: Strong call type specialization");
    } else {
        println!("✓ MODERATE VOCABULARY SHARING");
        println!("  → {:.0}% of words are context-specific",
                 context_specific_words as f64 / global_vocabulary.len() as f64 * 100.0);
        println!("  → Suggests: Shared acoustic elements across call types");
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Analysis complete!");
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
