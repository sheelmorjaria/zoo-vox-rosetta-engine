// Egyptian Fruit Bat Within-Vocalization Syntax Discovery
// ========================================================
//
// This example implements a comprehensive pipeline to discover syntactic structure
// within Egyptian fruit bat vocalizations by:
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
// 4. **Sequential Pattern Analysis**: Discovering syntactic rules through phrase ordering
//
// Dataset: /mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/
// - 91,080 vocalizations
// - 250kHz sample rate
// - 13 behavioral contexts (coded 0-12 in annotations.csv)

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::{
    hdbscan::HdbscanClustering, MicroDynamicsExtractor, WithinVocalizationAnalyzer,
    WithinVocalizationConfig,
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Bat behavioral context (from annotations.csv)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BatContext {
    Context0,
    Context1,
    Context2,
    Context3,
    Context4,
    Context5,
    Context6,
    Context7,
    Context8,
    Context9,
    Context10,
    Context11,
    Context12,
}

impl BatContext {
    fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(BatContext::Context0),
            1 => Some(BatContext::Context1),
            2 => Some(BatContext::Context2),
            3 => Some(BatContext::Context3),
            4 => Some(BatContext::Context4),
            5 => Some(BatContext::Context5),
            6 => Some(BatContext::Context6),
            7 => Some(BatContext::Context7),
            8 => Some(BatContext::Context8),
            9 => Some(BatContext::Context9),
            10 => Some(BatContext::Context10),
            11 => Some(BatContext::Context11),
            12 => Some(BatContext::Context12),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            BatContext::Context0 => "Context_0",
            BatContext::Context1 => "Context_1",
            BatContext::Context2 => "Context_2",
            BatContext::Context3 => "Context_3",
            BatContext::Context4 => "Context_4",
            BatContext::Context5 => "Context_5",
            BatContext::Context6 => "Context_6",
            BatContext::Context7 => "Context_7",
            BatContext::Context8 => "Context_8",
            BatContext::Context9 => "Context_9",
            BatContext::Context10 => "Context_10",
            BatContext::Context11 => "Context_11",
            BatContext::Context12 => "Context_12",
        }
    }
}

/// Phrase candidate extracted from within-vocalization segmentation
#[derive(Debug, Clone)]
struct PhraseCandidate {
    phrase_id: String,
    vocalization_file: String,
    context: BatContext,
    start_ms: f64,
    end_ms: f64,
    duration_ms: f64,
    features: Vec<f32>,
    f0_hz: Option<f64>,
    boundary_type: String,
}

/// Vocabulary word discovered through clustering
#[derive(Debug, Clone)]
struct VocabWord {
    word_id: usize,
    representative_features: Vec<f32>,
    member_phrases: Vec<String>,
    contexts: HashSet<BatContext>,
    source_vocalizations: HashSet<String>,
    occurrence_count: usize,
}

/// Reuse analysis results
#[derive(Debug)]
struct ReuseAnalysis {
    general_purpose_words: Vec<GeneralPurposeWord>,
    context_specific_words: Vec<ContextSpecificWord>,
    context_statistics: HashMap<String, ContextStats>,
}

#[derive(Debug)]
struct GeneralPurposeWord {
    word_id: usize,
    num_contexts: usize,
    occurrence_count: usize,
    num_vocalizations: usize,
    contexts: Vec<String>,
}

#[derive(Debug)]
struct ContextSpecificWord {
    word_id: usize,
    context: String,
    occurrence_count: usize,
    num_vocalizations: usize,
}

#[derive(Debug)]
struct ContextStats {
    total_phrases: usize,
    unique_words: usize,
    general_purpose_words: usize,
    context_specific_words: usize,
}

/// Sequential pattern analysis results
#[derive(Debug, Clone)]
struct SequentialAnalysis {
    bigram_counts: HashMap<String, usize>,
    trigram_counts: HashMap<String, usize>,
    top_transitions: Vec<(String, String, usize)>,
    context_sequences: HashMap<String, ContextSequences>,
    cross_context_patterns: Vec<CrossContextPattern>,
}

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
    pattern: Vec<String>,
    contexts: Vec<String>,
    occurrences: usize,
    pattern_type: String,
}

// ============================================================================
// AUDIO LOADING
// ============================================================================

/// Load a single WAV file and return audio samples
fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

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
                let n_frames = buf.frames();
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    for &sample in samples.iter() {
                        audio_samples.push(sample);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                let n_frames = buf.frames();
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    for &sample in samples.iter() {
                        audio_samples.push(sample as f32 / i16::MAX as f32);
                    }
                }
            }
            _ => {
                return Err("Unsupported audio format".into());
            }
        }
    }

    Ok(audio_samples)
}

/// Load annotations CSV and create file -> context mapping
fn load_annotations(
    annotations_path: &Path,
) -> Result<HashMap<String, BatContext>, Box<dyn std::error::Error>> {
    let mut mapping = HashMap::new();

    let content = std::fs::read_to_string(annotations_path)?;
    let lines: Vec<&str> = content.lines().collect();

    println!(
        "📊 Loading annotations from: {}",
        annotations_path.display()
    );

    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let file_name = parts[7].trim();
            if let Ok(context_code) = parts[2].trim().parse::<u32>() {
                if let Some(context) = BatContext::from_code(context_code) {
                    mapping.insert(file_name.to_string(), context);
                }
            }
        }
    }

    println!("  ✅ Loaded {} file → context mappings", mapping.len());
    Ok(mapping)
}

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

/// Extract features from audio segment (using 19D bat-optimized features)
fn extract_15d_features(
    audio: &[f32],
    sample_rate: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    // Use 19D RFE-optimal features for bats
    let features = extractor.extract_rfe_optimal_19d_bat(audio)?;
    // Return first 15D if available
    let feature_dim = features.len().min(15);
    Ok(features[..feature_dim].to_vec())
}

// ============================================================================
// PHASE 1: WITHIN-VOCALIZATION SEGMENTATION
// ============================================================================

fn segment_vocalization(
    audio: &[f32],
    file_name: &str,
    context: BatContext,
    analyzer: &WithinVocalizationAnalyzer,
    sample_rate: u32,
) -> Result<Vec<PhraseCandidate>, Box<dyn std::error::Error>> {
    let segmentation = analyzer.analyze_vocalization(audio, None)?;

    let mut phrases = Vec::new();

    for (i, (&start_ms, &duration_ms)) in segmentation
        .phrase_starts_ms
        .iter()
        .zip(segmentation.phrase_durations_ms.iter())
        .enumerate()
    {
        let end_ms = start_ms + duration_ms;

        // Skip very short phrases
        if duration_ms < 2.0 {
            continue;
        }

        let start_sample = (start_ms * sample_rate as f64 / 1000.0) as usize;
        let end_sample = (end_ms * sample_rate as f64 / 1000.0) as usize;

        if start_sample >= audio.len() || end_sample > audio.len() || start_sample >= end_sample {
            continue;
        }

        let phrase_audio = &audio[start_sample..end_sample];

        let features = extract_15d_features(phrase_audio, sample_rate)?;

        let phrase_id = format!("{} phrase_{:04}", file_name, i);

        // Determine boundary type
        let boundary_type = if i == 0 {
            "start".to_string()
        } else {
            "continuation".to_string()
        };

        phrases.push(PhraseCandidate {
            phrase_id,
            vocalization_file: file_name.to_string(),
            context,
            start_ms,
            end_ms,
            duration_ms,
            features,
            f0_hz: None, // F0 tracking not implemented
            boundary_type,
        });
    }

    Ok(phrases)
}

// ============================================================================
// PHASE 2: CROSS-VOCALIZATION CLUSTERING (HDBSCAN)
// ============================================================================

fn discover_vocabulary(
    phrases: &[PhraseCandidate],
    min_cluster_size: usize,
    min_samples: usize,
) -> Vec<VocabWord> {
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

    // Build feature matrix
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
            cluster_map
                .entry(label)
                .or_insert_with(Vec::new)
                .push(&phrases[i]);
        }
    }

    println!("  ✅ Found {} clusters", cluster_map.len());

    cluster_map
        .into_iter()
        .map(|(cluster_id, cluster)| {
            let word_id = cluster_id as usize;
            let n_features = cluster[0].features.len();
            let mut centroid = vec![0.0f32; n_features];

            let mut member_phrases: Vec<String> = Vec::new();
            let mut contexts: HashSet<BatContext> = HashSet::new();
            let mut source_vocalizations: HashSet<String> = HashSet::new();

            for phrase in &cluster {
                for (i, &val) in phrase.features.iter().enumerate() {
                    centroid[i] += val;
                }
                member_phrases.push(phrase.phrase_id.clone());
                contexts.insert(phrase.context);
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

/// Create a single word from all phrases (fallback when clustering fails)
fn create_single_word_from_phrases(phrases: &[PhraseCandidate], word_id: usize) -> VocabWord {
    let n_features = phrases[0].features.len();
    let mut centroid = vec![0.0f32; n_features];

    let mut member_phrases: Vec<String> = Vec::new();
    let mut contexts: HashSet<BatContext> = HashSet::new();
    let mut source_vocalizations: HashSet<String> = HashSet::new();

    for phrase in phrases {
        for (i, &val) in phrase.features.iter().enumerate() {
            centroid[i] += val;
        }
        member_phrases.push(phrase.phrase_id.clone());
        contexts.insert(phrase.context);
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

// ============================================================================
// PHASE 3: REUSE PATTERN ANALYSIS
// ============================================================================

fn analyze_reuse_patterns(vocabulary: &[VocabWord], phrases: &[PhraseCandidate]) -> ReuseAnalysis {
    println!("📊 Analyzing phrase reuse patterns...");

    // Create phrase -> word mapping
    let mut phrase_to_word: HashMap<String, usize> = HashMap::new();
    for word in vocabulary {
        for phrase_id in &word.member_phrases {
            phrase_to_word.insert(phrase_id.clone(), word.word_id);
        }
    }

    // Separate general purpose and context-specific words
    let mut general_purpose_words: Vec<GeneralPurposeWord> = Vec::new();
    let mut context_specific_words: Vec<ContextSpecificWord> = Vec::new();

    for word in vocabulary {
        let word_id = word.word_id;
        let occurrence_count = word.occurrence_count;
        let num_vocalizations = word.source_vocalizations.len();
        let contexts: Vec<String> = word.contexts.iter().map(|c| c.name().to_string()).collect();

        if word.contexts.len() >= 2 {
            general_purpose_words.push(GeneralPurposeWord {
                word_id,
                num_contexts: word.contexts.len(),
                occurrence_count,
                num_vocalizations,
                contexts,
            });
        } else {
            let context = word
                .contexts
                .iter()
                .next()
                .map(|c| c.name().to_string())
                .unwrap_or_default();
            context_specific_words.push(ContextSpecificWord {
                word_id,
                context,
                occurrence_count,
                num_vocalizations,
            });
        }
    }

    // Sort by occurrence count
    general_purpose_words.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
    context_specific_words.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

    println!(
        "  ✅ Found {} general purpose words",
        general_purpose_words.len()
    );
    println!(
        "  ✅ Found {} context-specific words",
        context_specific_words.len()
    );

    // Calculate context statistics
    let mut context_statistics: HashMap<String, ContextStats> = HashMap::new();

    // Group phrases by context
    let mut phrases_by_context: HashMap<BatContext, Vec<&PhraseCandidate>> = HashMap::new();
    for phrase in phrases {
        phrases_by_context
            .entry(phrase.context)
            .or_insert_with(Vec::new)
            .push(phrase);
    }

    for (context, context_phrases) in phrases_by_context {
        let context_name = context.name();
        let total_phrases = context_phrases.len();

        let mut unique_words: HashSet<usize> = HashSet::new();
        let mut general_count = 0;
        let mut specific_count = 0;

        for phrase in context_phrases {
            if let Some(&word_id) = phrase_to_word.get(&phrase.phrase_id) {
                unique_words.insert(word_id);

                if let Some(word) = vocabulary.get(word_id) {
                    if word.contexts.len() >= 2 {
                        general_count += 1;
                    } else {
                        specific_count += 1;
                    }
                }
            }
        }

        context_statistics.insert(
            context_name.to_string(),
            ContextStats {
                total_phrases,
                unique_words: unique_words.len(),
                general_purpose_words: general_count,
                context_specific_words: specific_count,
            },
        );
    }

    ReuseAnalysis {
        general_purpose_words,
        context_specific_words,
        context_statistics,
    }
}

// ============================================================================
// PHASE 4: SEQUENTIAL PATTERN ANALYSIS
// ============================================================================

fn analyze_sequential_patterns(phrases: &[PhraseCandidate]) -> SequentialAnalysis {
    // Group phrases by vocalization
    let mut vocalization_phrases: HashMap<String, Vec<&PhraseCandidate>> = HashMap::new();
    for phrase in phrases {
        vocalization_phrases
            .entry(phrase.vocalization_file.clone())
            .or_insert_with(Vec::new)
            .push(phrase);
    }

    // Sort phrases within each vocalization by start time
    for phrases_vec in vocalization_phrases.values_mut() {
        phrases_vec.sort_by(|a, b| {
            a.start_ms
                .partial_cmp(&b.start_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

    // Group vocalizations by context (use HashSet to track unique vocalizations per context)
    let mut context_vocalizations: HashMap<BatContext, HashSet<String>> = HashMap::new();
    for phrase in phrases {
        context_vocalizations
            .entry(phrase.context)
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
                    sequence_lengths.iter().map(|&x| x as f64).sum::<f64>()
                        / sequence_lengths.len() as f64
                },
                sequence_lengths,
                bigram_entropy,
                most_common_bigrams,
            },
        );
    }

    // Find cross-context patterns (shared bigrams/trigrams)
    let mut cross_context_patterns: Vec<CrossContextPattern> = Vec::new();

    // Re-extract with context tracking
    let mut context_bigrams: HashMap<BatContext, HashSet<String>> = HashMap::new();
    for phrases_vec in vocalization_phrases.values() {
        if let Some(first_phrase) = phrases_vec.first() {
            let context = first_phrase.context;
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
    let all_bigrams: HashSet<String> = context_bigrams
        .values()
        .flat_map(|set| set.iter().cloned())
        .collect();

    for bigram in all_bigrams {
        let mut contexts: Vec<BatContext> = Vec::new();
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
    println!(
        "  ✅ Found {} cross-context patterns",
        cross_context_patterns.len()
    );

    SequentialAnalysis {
        bigram_counts,
        trigram_counts,
        top_transitions,
        context_sequences,
        cross_context_patterns,
    }
}

fn calculate_entropy(items: &[String]) -> f64 {
    if items.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for item in items {
        *counts.entry(item).or_insert(0) += 1;
    }

    let total = items.len() as f64;
    let mut entropy = 0.0;

    for &count in counts.values() {
        let probability = count as f64 / total;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }

    entropy
}

// ============================================================================
// PHASE 4 DISPLAY
// ============================================================================

fn run_sequential_analysis(
    phrases: &[PhraseCandidate],
) -> Result<SequentialAnalysis, Box<dyn std::error::Error>> {
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
    for (i, (from_phrase, to_phrase, count)) in analysis.top_transitions.iter().take(20).enumerate()
    {
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

    for (context_name, seq_data) in context_entries.iter().take(13) {
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
            println!(
                "  {:2}. Pattern: {}",
                i + 1,
                truncate_string(&pattern_str, 50)
            );
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
    let all_entropies: Vec<f64> = analysis
        .context_sequences
        .values()
        .map(|s| s.bigram_entropy)
        .collect();

    if !all_entropies.is_empty() {
        let avg_entropy = all_entropies.iter().sum::<f64>() / all_entropies.len() as f64;
        let min_entropy = all_entropies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_entropy = all_entropies
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!(
            "Average bigram entropy (unpredictability): {:.3} bits",
            avg_entropy
        );
        println!(
            "Entropy range: [{:.3}, {:.3}] bits",
            min_entropy, max_entropy
        );
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
    println!(
        "Cross-context patterns: {}",
        analysis.cross_context_patterns.len()
    );
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

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let base_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = base_dir.join("audio");
    let annotations_path = base_dir.join("annotations.csv");

    let mut limit_vocalizations: Option<usize> = None;
    let mut min_cluster_size: Option<usize> = None;
    let mut min_samples: Option<usize> = None;
    let mut skip_clustering = false;

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                if i + 1 < args.len() {
                    limit_vocalizations = Some(args[i + 1].parse()?);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--min-cluster-size" => {
                if i + 1 < args.len() {
                    min_cluster_size = Some(args[i + 1].parse()?);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--min-samples" => {
                if i + 1 < args.len() {
                    min_samples = Some(args[i + 1].parse()?);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--skip-clustering" => {
                skip_clustering = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Egyptian Fruit Bat Within-Vocalization Syntax Discovery             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    if let Some(limit) = limit_vocalizations {
        println!("📊 Limiting to {} files for testing", limit);
        println!(
            "📊 Processing limited to {} vocalizations (testing mode)",
            limit
        );
    }

    println!("📂 Base directory: {}", base_dir.display());
    println!("🎵 Audio directory: {}", audio_dir.display());
    println!("📋 Annotations: {}", annotations_path.display());
    println!();

    // Check if directories exist
    if !audio_dir.exists() {
        println!("⚠️  Audio directory not found: {}", audio_dir.display());
        return Err("Audio directory not found".into());
    }

    if !annotations_path.exists() {
        println!(
            "⚠️  Annotations file not found: {}",
            annotations_path.display()
        );
        return Err("Annotations file not found".into());
    }

    // Load annotations
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading Annotations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let file_to_context = load_annotations(&annotations_path)?;

    // Scan for audio files
    println!("📂 Scanning audio directory: {}", audio_dir.display());

    let audio_files: Vec<PathBuf> = std::fs::read_dir(&audio_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();

    println!("✅ Discovered {} audio files", audio_files.len());

    if let Some(limit) = limit_vocalizations {
        println!("📊 Limited to {} files (was {})", limit, audio_files.len());
    }

    println!();

    // =========================================================================
    // PHASE 1: WITHIN-VOCALIZATION SEGMENTATION
    // =========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 1: Within-Vocalization Phrase Segmentation                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Configuration for bat vocalizations (250kHz, high-frequency detection)
    let config = WithinVocalizationConfig {
        sample_rate: 250000,
        min_phrase_duration_ms: 5.0,
        min_pause_duration_ms: 2.0,
        min_f0_change_hz: 1500.0,
        pause_energy_threshold: 0.15,
        frame_size_ms: 2.0,
        hop_size_ms: 1.0,
        require_consensus: false,
        max_phrases: 8,
    };

    println!("Configuration:");
    println!("  - Sample rate: {} kHz", config.sample_rate / 1000);
    println!(
        "  - Min phrase duration: {} ms",
        config.min_phrase_duration_ms
    );
    println!(
        "  - Min pause duration: {} ms",
        config.min_pause_duration_ms
    );
    println!("  - Min F0 change: {} Hz", config.min_f0_change_hz);
    println!();

    let analyzer = WithinVocalizationAnalyzer::new(config.clone());

    let progress = Arc::new(AtomicUsize::new(0));
    let total = if let Some(limit) = limit_vocalizations {
        limit
    } else {
        audio_files.len()
    };

    let mut phrases: Vec<PhraseCandidate> = Vec::new();
    let start_time = std::time::Instant::now();

    // Process vocalizations
    let files_to_process: Vec<_> = if let Some(limit) = limit_vocalizations {
        audio_files.iter().take(limit).collect()
    } else {
        audio_files.iter().collect()
    };

    for (i, audio_file) in files_to_process.iter().enumerate() {
        let file_name = audio_file
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Get context from annotations
        let context = file_to_context
            .get(&file_name)
            .copied()
            .unwrap_or(BatContext::Context0);

        match load_wav_file(audio_file) {
            Ok(audio) => {
                match segment_vocalization(
                    &audio,
                    &file_name,
                    context,
                    &analyzer,
                    config.sample_rate,
                ) {
                    Ok(mut phrase_list) => {
                        phrases.append(&mut phrase_list);
                    }
                    Err(e) => {
                        eprintln!("  ✗ Error segmenting {}: {}", file_name, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  ✗ Error loading {}: {}", file_name, e);
            }
        }

        progress.fetch_add(1, Ordering::Relaxed);

        if i % 100 == 0 || i == files_to_process.len() - 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = progress.load(Ordering::Relaxed) as f64 / elapsed;
            let remaining = ((total - progress.load(Ordering::Relaxed)) as f64 / rate) as u64;
            println!(
                "  Progress: {}/{} ({:.1}%) | {:.1} items/sec | ETA: {}s",
                progress.load(Ordering::Relaxed),
                total,
                100.0 * progress.load(Ordering::Relaxed) as f64 / total as f64,
                rate,
                remaining
            );
        }
    }

    println!("  ✅ Extracted {} phrase candidates", phrases.len());
    println!();

    if phrases.is_empty() {
        println!("⚠️  No phrase candidates extracted. Cannot continue.");
        return Ok(());
    }

    // =========================================================================
    // PHASE 2: CROSS-VOCALIZATION CLUSTERING
    // =========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 2: Cross-Vocalization Phrase Clustering                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let vocabulary = if skip_clustering {
        println!(
            "⏭ Skip-clustering mode: treating each of {} phrases as its own word",
            phrases.len()
        );
        phrases
            .iter()
            .enumerate()
            .map(|(i, phrase)| {
                let mut contexts = HashSet::new();
                contexts.insert(phrase.context);
                VocabWord {
                    word_id: i,
                    representative_features: phrase.features.clone(),
                    member_phrases: vec![phrase.phrase_id.clone()],
                    contexts,
                    source_vocalizations: {
                        let mut set = HashSet::new();
                        set.insert(phrase.vocalization_file.clone());
                        set
                    },
                    occurrence_count: 1,
                }
            })
            .collect()
    } else {
        let min_cluster = min_cluster_size.unwrap_or_else(|| {
            // Auto-calculate: ~2% of phrases, minimum 5
            let suggested = (phrases.len() as f64 * 0.02).ceil() as usize;
            suggested.max(5)
        });

        let min_samples_val = min_samples.unwrap_or_else(|| {
            // Auto-calculate: ~1% of phrases, minimum 3
            let suggested = (phrases.len() as f64 * 0.01).ceil() as usize;
            suggested.max(3)
        });

        discover_vocabulary(&phrases, min_cluster, min_samples_val)
    };

    println!();
    println!("Global Vocabulary Statistics:");
    println!("  Total vocabulary size: {} words", vocabulary.len());
    println!();

    // =========================================================================
    // PHASE 3: REUSE PATTERN ANALYSIS
    // =========================================================================

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
    for (i, word) in reuse_analysis
        .general_purpose_words
        .iter()
        .take(20)
        .enumerate()
    {
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
    let mut context_specific_by_context: HashMap<String, Vec<&ContextSpecificWord>> =
        HashMap::new();
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
