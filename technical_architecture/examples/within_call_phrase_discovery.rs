// Within-Call Phrase Discovery using Acoustic Similarity Engine
//
// This example demonstrates how to analyze the INTERNAL structure of individual
// vocalizations to discover:
// - Repeated phrase types within a single call
// - Phrase sequences and potential "syntax"
// - Motif patterns (repeated phrase sequences)
// - Individual vocal repertoire
//
// KEY INSIGHT: Uses SIMILARITY THRESHOLDING instead of clustering.
// This respects the continuous acoustic manifold nature of vocalizations.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use technical_architecture::{
    AcousticSimilarityEngine,
    MicroDynamicsExtractor,
};

// =============================================================================
// Configuration
// =============================================================================

/// Segmentation configuration for phrase extraction
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Minimum phrase duration in milliseconds
    pub min_phrase_ms: f64,

    /// Maximum phrase duration in milliseconds
    pub max_phrase_ms: f64,

    /// Energy threshold for silence detection (relative to max)
    pub energy_threshold: f64,

    /// Minimum gap between phrases in milliseconds
    pub min_gap_ms: f64,

    /// Use fixed window size instead of adaptive (None = adaptive)
    pub fixed_window_ms: Option<f64>,

    /// Sample rate
    pub sample_rate: u32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            min_phrase_ms: 30.0,
            max_phrase_ms: 500.0,
            energy_threshold: 0.05,
            min_gap_ms: 10.0,
            fixed_window_ms: None,
            sample_rate: 96000,
        }
    }
}

// =============================================================================
// Phrase Candidate
// =============================================================================

/// A candidate phrase extracted from a vocalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidate {
    /// Unique ID within the vocalization
    pub id: usize,

    /// Start time in milliseconds
    pub start_ms: f64,

    /// End time in milliseconds
    pub end_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Start sample index
    pub start_sample: usize,

    /// End sample index
    pub end_sample: usize,

    /// 30D feature vector
    pub features: Vec<f64>,

    /// Similarity-based phrase type (assigned later)
    pub phrase_type: Option<i32>,

    /// Confidence of phrase type assignment
    pub type_confidence: Option<f64>,
}

// =============================================================================
// Phrase Segmenter
// =============================================================================

pub struct PhraseSegmenter {
    config: SegmentationConfig,
}

impl PhraseSegmenter {
    pub fn new(config: SegmentationConfig) -> Self {
        Self { config }
    }

    /// Segment a vocalization into phrase candidates
    pub fn segment(&self, audio: &[f32]) -> Vec<PhraseCandidate> {
        self.segment_adaptive(audio)
    }

    /// Adaptive segmentation based on energy gaps
    fn segment_adaptive(&self, audio: &[f32]) -> Vec<PhraseCandidate> {
        let n = audio.len();
        let sample_rate = self.config.sample_rate as f64;

        if n == 0 {
            return vec![];
        }

        // Compute RMS energy in windows
        let window_samples = (sample_rate * 0.005) as usize; // 5ms windows
        if window_samples == 0 {
            return vec![];
        }
        let n_windows = n / window_samples;

        if n_windows == 0 {
            return vec![];
        }

        let mut energy_profile = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            let start = i * window_samples;
            let end = (start + window_samples).min(n);
            let rms: f32 = audio[start..end].iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt() / (end - start) as f32;
            energy_profile.push(rms);
        }

        // Normalize energy profile
        let max_energy = energy_profile.iter().cloned().fold(0.0f32, f32::max);
        if max_energy == 0.0 {
            return vec![];
        }
        let threshold = max_energy * self.config.energy_threshold as f32;

        // Find phrase boundaries
        let min_phrase_windows = (self.config.min_phrase_ms / 5.0) as usize;
        let min_gap_windows = (self.config.min_gap_ms / 5.0) as usize;

        let mut candidates = Vec::new();
        let mut in_phrase = false;
        let mut phrase_start = 0;
        let mut phrase_count = 0;
        let mut silence_count = 0;

        for (i, &energy) in energy_profile.iter().enumerate() {
            if energy >= threshold {
                if !in_phrase {
                    phrase_start = i;
                    in_phrase = true;
                    silence_count = 0;
                } else {
                    silence_count = 0;
                }
            } else {
                if in_phrase {
                    silence_count += 1;
                    if silence_count >= min_gap_windows {
                        let phrase_end = i.saturating_sub(silence_count);
                        let phrase_len = phrase_end.saturating_sub(phrase_start);

                        if phrase_len >= min_phrase_windows {
                            let start_sample = phrase_start * window_samples;
                            let end_sample = phrase_end * window_samples;

                            candidates.push(PhraseCandidate {
                                id: phrase_count,
                                start_ms: start_sample as f64 / sample_rate * 1000.0,
                                end_ms: end_sample as f64 / sample_rate * 1000.0,
                                duration_ms: (end_sample - start_sample) as f64 / sample_rate * 1000.0,
                                start_sample,
                                end_sample,
                                features: Vec::new(),
                                phrase_type: None,
                                type_confidence: None,
                            });

                            phrase_count += 1;
                        }

                        in_phrase = false;
                    }
                }
            }
        }

        // Handle final phrase
        if in_phrase {
            let phrase_end = energy_profile.len().saturating_sub(silence_count);
            let phrase_len = phrase_end.saturating_sub(phrase_start);

            if phrase_len >= min_phrase_windows {
                let start_sample = phrase_start * window_samples;
                let end_sample = (phrase_end * window_samples).min(n);

                candidates.push(PhraseCandidate {
                    id: phrase_count,
                    start_ms: start_sample as f64 / sample_rate * 1000.0,
                    end_ms: end_sample as f64 / sample_rate * 1000.0,
                    duration_ms: (end_sample - start_sample) as f64 / sample_rate * 1000.0,
                    start_sample,
                    end_sample,
                    features: Vec::new(),
                    phrase_type: None,
                    type_confidence: None,
                });
            }
        }

        // Filter by max duration and renumber
        candidates.into_iter()
            .filter(|c| c.duration_ms <= self.config.max_phrase_ms)
            .enumerate()
            .map(|(i, mut c)| {
                c.id = i;
                c
            })
            .collect()
    }
}

// =============================================================================
// Motif Discovery
// =============================================================================

/// A discovered motif (repeated phrase sequence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    pub id: usize,
    pub pattern: Vec<i32>,
    pub occurrences: usize,
    pub positions: Vec<Vec<usize>>,
}

// =============================================================================
// Within-Call Analysis Results
// =============================================================================

/// Results of within-call phrase analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallAnalysis {
    pub file_name: String,
    pub call_type: Option<String>,
    pub total_duration_ms: f64,
    pub phrases: Vec<PhraseCandidate>,
    pub n_phrase_types: usize,
    pub phrase_types: Vec<i32>,
    pub similarity_matrix: Vec<Vec<f64>>,
    pub distance_matrix: Vec<Vec<f64>>,
    pub motifs: Vec<Motif>,
    pub phrase_sequence: Vec<i32>,
    pub transition_matrix: HashMap<i32, HashMap<i32, f64>>,
    pub stats: WithinCallStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallStats {
    pub n_phrases: usize,
    pub avg_phrase_duration_ms: f64,
    pub type_distribution: HashMap<i32, usize>,
    pub type_entropy: f64,
    pub avg_within_type_similarity: f64,
    pub avg_between_type_distance: f64,
    pub phrase_rate: f64,
}

// =============================================================================
// Within-Call Analyzer
// =============================================================================

pub struct WithinCallAnalyzer {
    segmenter: PhraseSegmenter,
    same_type_threshold: f64,
    feature_dim: usize,
}

impl WithinCallAnalyzer {
    pub fn new(config: SegmentationConfig, feature_dim: usize) -> Self {
        Self {
            segmenter: PhraseSegmenter::new(config),
            same_type_threshold: 0.75,
            feature_dim,
        }
    }

    /// Analyze a single vocalization
    pub fn analyze(
        &self,
        audio: &[f32],
        file_name: &str,
        call_type: Option<&str>,
    ) -> WithinCallAnalysis {
        // Step 1: Segment into phrases
        let mut phrases = self.segmenter.segment(audio);

        // Step 2: Extract features for each phrase
        let extractor = MicroDynamicsExtractor::new(self.segmenter.config.sample_rate);
        for phrase in &mut phrases {
            let phrase_audio = &audio[phrase.start_sample..phrase.end_sample];
            if let Ok(features) = extractor.extract(phrase_audio) {
                // Convert 30D features to Vec<f64>
                phrase.features = vec![
                    features.attack_time_ms as f64,
                    features.decay_time_ms as f64,
                    features.sustain_level as f64,
                    features.vibrato_rate_hz as f64,
                    features.vibrato_depth as f64,
                    features.jitter as f64,
                    features.shimmer as f64,
                    features.harmonicity as f64,
                    features.spectral_flatness as f64,
                    features.harmonic_to_noise_ratio as f64,
                ];
                // Add MFCCs
                phrase.features.extend(features.mfcc.iter().map(|&v| v as f64));
                // Add spectral_flux
                phrase.features.push(features.spectral_flux as f64);
                // Add rhythm features
                phrase.features.push(features.median_ici_ms as f64);
                phrase.features.push(features.onset_rate_hz as f64);
                phrase.features.push(features.ici_coefficient_of_variation as f64);
            }
        }

        let n = phrases.len();

        // Step 3: Compute pairwise similarity/distance matrix
        let mut similarity_matrix = vec![vec![0.0f64; n]; n];
        let mut distance_matrix = vec![vec![0.0f64; n]; n];

        let actual_dim = phrases.first().map(|p| p.features.len()).unwrap_or(self.feature_dim);
        let engine = AcousticSimilarityEngine::new(actual_dim);

        for i in 0..n {
            for j in i..n {
                let a = Array1::from_vec(phrases[i].features.clone());
                let b = Array1::from_vec(phrases[j].features.clone());

                let dist = engine.distance(&a, &b);
                let sim = engine.similarity(&a, &b);

                distance_matrix[i][j] = dist;
                distance_matrix[j][i] = dist;
                similarity_matrix[i][j] = sim;
                similarity_matrix[j][i] = sim;
            }
        }

        // Step 4: Discover phrase types
        let (phrase_types, n_phrase_types) = self.discover_phrase_types(&phrases, &distance_matrix);

        // Step 5: Assign types to phrases
        for (phrase, &ptype) in phrases.iter_mut().zip(phrase_types.iter()) {
            phrase.phrase_type = Some(ptype);

            let same_type: Vec<usize> = phrase_types.iter()
                .enumerate()
                .filter(|(_, &t)| t == ptype)
                .map(|(i, _)| i)
                .collect();

            if same_type.len() > 1 {
                let avg_sim: f64 = same_type.iter()
                    .filter(|&&idx| idx != phrase.id)
                    .map(|&idx| similarity_matrix[phrase.id][idx])
                    .sum::<f64>() / (same_type.len() - 1) as f64;
                phrase.type_confidence = Some(avg_sim);
            } else {
                phrase.type_confidence = Some(1.0);
            }
        }

        // Step 6: Discover motifs
        let motifs = self.discover_motifs(&phrase_types);

        // Step 7: Build transition matrix
        let transition_matrix = self.build_transition_matrix(&phrase_types);

        // Step 8: Compute statistics
        let stats = self.compute_stats(
            &phrases,
            &phrase_types,
            n_phrase_types,
            &similarity_matrix,
            &distance_matrix,
        );

        // Clone for sequence output
        let phrase_sequence = phrase_types.clone();

        WithinCallAnalysis {
            file_name: file_name.to_string(),
            call_type: call_type.map(|s| s.to_string()),
            total_duration_ms: audio.len() as f64 / self.segmenter.config.sample_rate as f64 * 1000.0,
            phrases,
            n_phrase_types,
            phrase_types,
            similarity_matrix,
            distance_matrix,
            motifs,
            phrase_sequence,
            transition_matrix,
            stats,
        }
    }

    fn discover_phrase_types(
        &self,
        phrases: &[PhraseCandidate],
        distance_matrix: &[Vec<f64>],
    ) -> (Vec<i32>, usize) {
        let n = phrases.len();
        if n == 0 {
            return (vec![], 0);
        }

        let mut phrase_types = vec![-1i32; n];
        let mut next_type = 0i32;

        for i in 0..n {
            if phrase_types[i] != -1 {
                continue;
            }

            // Check if similar to existing type
            let mut best_type = -1i32;
            let mut best_sim = 0.0f64;

            for j in 0..i {
                if phrase_types[j] != -1 {
                    let sim = 1.0 - (-distance_matrix[i][j]).exp();
                    if sim >= self.same_type_threshold && sim > best_sim {
                        best_sim = sim;
                        best_type = phrase_types[j];
                    }
                }
            }

            if best_type >= 0 {
                phrase_types[i] = best_type;
            } else {
                phrase_types[i] = next_type;

                for j in (i + 1)..n {
                    if phrase_types[j] == -1 {
                        let sim = 1.0 - (-distance_matrix[i][j]).exp();
                        if sim >= self.same_type_threshold {
                            phrase_types[j] = next_type;
                        }
                    }
                }

                next_type += 1;
            }
        }

        // Renumber types consecutively
        let mut type_map: HashMap<i32, i32> = HashMap::new();
        let mut new_id = 0i32;

        for t in &mut phrase_types {
            if let Some(&mapped) = type_map.get(t) {
                *t = mapped;
            } else {
                type_map.insert(*t, new_id);
                *t = new_id;
                new_id += 1;
            }
        }

        (phrase_types, new_id as usize)
    }

    fn discover_motifs(&self, phrase_types: &[i32]) -> Vec<Motif> {
        let n = phrase_types.len();
        if n < 3 {
            return vec![];
        }

        let mut motifs: Vec<Motif> = Vec::new();
        let mut motif_id = 0;

        for motif_len in 2..=5.min(n / 2) {
            let mut pattern_counts: HashMap<Vec<i32>, Vec<Vec<usize>>> = HashMap::new();

            for i in 0..=(n - motif_len) {
                let pattern: Vec<i32> = phrase_types[i..i + motif_len].to_vec();
                pattern_counts
                    .entry(pattern.clone())
                    .or_default()
                    .push((i..i + motif_len).collect());
            }

            for (pattern, positions) in pattern_counts {
                if positions.len() >= 2 {
                    motifs.push(Motif {
                        id: motif_id,
                        pattern,
                        occurrences: positions.len(),
                        positions,
                    });
                    motif_id += 1;
                }
            }
        }

        motifs.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
        motifs
    }

    fn build_transition_matrix(&self, phrase_types: &[i32]) -> HashMap<i32, HashMap<i32, f64>> {
        let mut transitions: HashMap<i32, HashMap<i32, usize>> = HashMap::new();
        let mut type_counts: HashMap<i32, usize> = HashMap::new();

        for window in phrase_types.windows(2) {
            let from = window[0];
            let to = window[1];

            *transitions.entry(from).or_default().entry(to).or_default() += 1;
            *type_counts.entry(from).or_default() += 1;
        }

        let mut prob_matrix: HashMap<i32, HashMap<i32, f64>> = HashMap::new();

        for (from, to_counts) in transitions {
            let total = type_counts.get(&from).copied().unwrap_or(1) as f64;
            let probs: HashMap<i32, f64> = to_counts.into_iter()
                .map(|(to, count)| (to, count as f64 / total))
                .collect();
            prob_matrix.insert(from, probs);
        }

        prob_matrix
    }

    fn compute_stats(
        &self,
        phrases: &[PhraseCandidate],
        phrase_types: &[i32],
        _n_phrase_types: usize,
        similarity_matrix: &[Vec<f64>],
        distance_matrix: &[Vec<f64>],
    ) -> WithinCallStats {
        let n_phrases = phrases.len();

        let mut type_distribution: HashMap<i32, usize> = HashMap::new();
        for &t in phrase_types {
            *type_distribution.entry(t).or_default() += 1;
        }

        let type_entropy = if n_phrases > 0 {
            let mut entropy = 0.0;
            for &count in type_distribution.values() {
                let p = count as f64 / n_phrases as f64;
                if p > 0.0 {
                    entropy -= p * p.log2();
                }
            }
            entropy
        } else {
            0.0
        };

        let avg_within_type_similarity = self.compute_within_type_similarity(
            phrase_types, similarity_matrix
        );

        let avg_between_type_distance = self.compute_between_type_distance(
            phrase_types, distance_matrix
        );

        let total_duration_ms: f64 = phrases.iter()
            .map(|p| p.duration_ms)
            .sum();
        let phrase_rate = if total_duration_ms > 0.0 {
            n_phrases as f64 / (total_duration_ms / 1000.0)
        } else {
            0.0
        };

        WithinCallStats {
            n_phrases,
            avg_phrase_duration_ms: if n_phrases > 0 {
                phrases.iter().map(|p| p.duration_ms).sum::<f64>() / n_phrases as f64
            } else {
                0.0
            },
            type_distribution,
            type_entropy,
            avg_within_type_similarity,
            avg_between_type_distance,
            phrase_rate,
        }
    }

    fn compute_within_type_similarity(
        &self,
        phrase_types: &[i32],
        similarity_matrix: &[Vec<f64>],
    ) -> f64 {
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..phrase_types.len() {
            for j in (i + 1)..phrase_types.len() {
                if phrase_types[i] == phrase_types[j] {
                    total_sim += similarity_matrix[i][j];
                    count += 1;
                }
            }
        }

        if count > 0 { total_sim / count as f64 } else { 1.0 }
    }

    fn compute_between_type_distance(
        &self,
        phrase_types: &[i32],
        distance_matrix: &[Vec<f64>],
    ) -> f64 {
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..phrase_types.len() {
            for j in (i + 1)..phrase_types.len() {
                if phrase_types[i] != phrase_types[j] {
                    total_dist += distance_matrix[i][j];
                    count += 1;
                }
            }
        }

        if count > 0 { total_dist / count as f64 } else { 0.0 }
    }
}

impl WithinCallAnalysis {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║         Within-Call Phrase Analysis Summary                    ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📁 File: {}", self.file_name);
        if let Some(ct) = &self.call_type {
            println!("   Call type: {}", ct);
        }
        println!("   Duration: {:.1} ms", self.total_duration_ms);

        println!("\n📊 Phrase Statistics:");
        println!("   • Total phrases: {}", self.stats.n_phrases);
        println!("   • Unique phrase types: {}", self.n_phrase_types);
        println!("   • Average phrase duration: {:.1} ms", self.stats.avg_phrase_duration_ms);
        println!("   • Phrase rate: {:.2} phrases/sec", self.stats.phrase_rate);
        println!("   • Type entropy: {:.3} bits", self.stats.type_entropy);

        println!("\n📊 Phrase Type Distribution:");
        let mut dist: Vec<_> = self.stats.type_distribution.iter().collect();
        dist.sort_by(|a, b| b.1.cmp(a.1));
        for (ptype, count) in dist.iter() {
            let pct = **count as f64 / self.stats.n_phrases.max(1) as f64 * 100.0;
            println!("   • Type {}: {} occurrences ({:.1}%)", ptype, count, pct);
        }

        println!("\n📊 Phrase Sequence:");
        let seq_str: String = self.phrase_sequence.iter()
            .map(|t| format!("{}", t))
            .collect::<Vec<_>>()
            .join(" → ");
        println!("   {}", seq_str);

        if !self.motifs.is_empty() {
            println!("\n📊 Discovered Motifs:");
            for motif in self.motifs.iter().take(5) {
                let pattern_str: String = motif.pattern.iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join("-");
                println!("   • Pattern [{}]: {} occurrences", pattern_str, motif.occurrences);
            }
        }

        println!("\n📊 Similarity Metrics:");
        println!("   • Avg within-type similarity: {:.4}", self.stats.avg_within_type_similarity);
        println!("   • Avg between-type distance: {:.4}", self.stats.avg_between_type_distance);
    }
}

// =============================================================================
// Audio Loading
// =============================================================================

fn load_flac_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
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
    let sample_rate = decoder.codec_params().sample_rate.unwrap_or(48000);

    let mut audio_samples = Vec::new();
    let mut packet_count = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };
        packet_count += 1;

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
                    audio_samples.extend(samples.iter().map(|s| s.0 as f32 / 8_388_607.0));
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
            AudioBufferRef::U16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
                }
            }
            AudioBufferRef::U24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|s| (s.0 as f32 - 8_388_608.0) / 8_388_608.0));
                }
            }
            AudioBufferRef::U32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 2_147_483_648.0) / 2_147_483_648.0));
                }
            }
            AudioBufferRef::F64(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32));
                }
            }
            AudioBufferRef::S8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i8::MAX as f32));
                }
            }
        }
    }

    // Store sample rate as metadata (caller can use it)
    let _ = (sample_rate, packet_count);

    Ok(audio_samples)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║    Within-Call Phrase Discovery using Similarity Engine        ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  🔍 Analyzes INTERNAL structure of individual vocalizations    ║");
    println!("║  📊 Discovers repeated phrases, motifs, and \"syntax\"           ║");
    println!("║  🎯 Uses SIMILARITY THRESHOLDING (not clustering)              ║");
    println!("║                                                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let home = std::env::var("HOME").unwrap_or_else(|_| "/mnt/c/Users/sheel".to_string());
    let vocalizations_dir = PathBuf::from(format!("{}/birdsong_analysis/data/Vocalizations", home));
    let output_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/within_call_results");

    if !vocalizations_dir.exists() {
        println!("❌ Vocalizations directory not found: {}", vocalizations_dir.display());
        return Ok(());
    }

    fs::create_dir_all(&output_dir)?;

    // Find FLAC files recursively
    let mut flac_files: Vec<PathBuf> = Vec::new();
    fn find_flac_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                find_flac_files(&path, files)?;
            } else if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "flac" {
                    files.push(path);
                }
            }
        }
        Ok(())
    }
    find_flac_files(&vocalizations_dir, &mut flac_files)?;

    let total_files = flac_files.len();
    println!("📂 Found {} FLAC files to process", total_files);
    println!();

    if total_files == 0 {
        println!("❌ No FLAC files found!");
        return Ok(());
    }

    // Configure analyzer
    let config = SegmentationConfig {
        min_phrase_ms: 30.0,
        max_phrase_ms: 500.0,
        energy_threshold: 0.05,
        min_gap_ms: 10.0,
        sample_rate: 96000,
        ..Default::default()
    };

    // Progress tracking with atomics for parallel processing
    let processed = AtomicUsize::new(0);
    let errors_count = AtomicUsize::new(0);
    let phrases_count = AtomicUsize::new(0);
    let types_count = AtomicUsize::new(0);
    let start_time = Instant::now();

    // Checkpoint interval
    let checkpoint_interval = if total_files > 10000 { 10000 } else { 1000 };
    let all_analyses: Mutex<Vec<WithinCallAnalysis>> = Mutex::new(Vec::new());

    println!("🚀 Starting PARALLEL analysis with {} threads...", rayon::current_num_threads());
    println!("   Total files: {}", total_files);
    println!("   Checkpoints every {} files", checkpoint_interval);
    println!();

    // Process files in parallel batches for better progress tracking
    let batch_size = 1000;
    let num_batches = (total_files + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(total_files);
        let batch: Vec<_> = flac_files[start..end].to_vec();

        let results: Vec<Option<WithinCallAnalysis>> = batch
            .par_iter()
            .map(|path| {
                let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
                let analyzer = WithinCallAnalyzer::new(config.clone(), 30);

                match load_flac_file(path) {
                    Ok(audio) => {
                        let analysis = analyzer.analyze(&audio, filename, None);
                        phrases_count.fetch_add(analysis.stats.n_phrases, Ordering::Relaxed);
                        types_count.fetch_add(analysis.n_phrase_types, Ordering::Relaxed);
                        processed.fetch_add(1, Ordering::Relaxed);
                        Some(analysis)
                    }
                    Err(_) => {
                        errors_count.fetch_add(1, Ordering::Relaxed);
                        processed.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                }
            })
            .collect();

        // Collect results
        {
            let mut analyses = all_analyses.lock().unwrap();
            for result in results {
                if let Some(analysis) = result {
                    analyses.push(analysis);
                }
            }
        }

        // Progress report
        let processed_val = processed.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed().as_secs_f64();
        let rate = processed_val as f64 / elapsed;
        let remaining = (total_files - processed_val) as f64 / rate;
        let phrases = phrases_count.load(Ordering::Relaxed);
        let errors = errors_count.load(Ordering::Relaxed);

        println!("   [{}/{}] {:.1}% | {:.0} files/min | ETA: {:.0}s | Phrases: {} | Errors: {}",
                 processed_val, total_files,
                 processed_val as f64 / total_files as f64 * 100.0,
                 rate * 60.0,
                 remaining,
                 phrases,
                 errors);

        // Checkpoint save
        if (batch_idx + 1) * batch_size >= checkpoint_interval &&
           ((batch_idx + 1) * batch_size) % checkpoint_interval == 0 {
            let checkpoint_path = output_dir.join(format!("checkpoint_{}.json", processed_val));
            let analyses = all_analyses.lock().unwrap();
            if let Ok(json) = serde_json::to_string_pretty(&*analyses) {
                let _ = fs::write(&checkpoint_path, json);
                println!("   💾 Checkpoint saved: {} analyses", analyses.len());
            }
        }
    }

    let all_analyses = all_analyses.into_inner().unwrap();

    // Aggregate statistics
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    AGGREGATE STATISTICS                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let total_phrases: usize = all_analyses.iter().map(|a| a.stats.n_phrases).sum();
    let total_types: usize = all_analyses.iter().map(|a| a.n_phrase_types).sum();
    let avg_entropy: f64 = all_analyses.iter()
        .map(|a| a.stats.type_entropy)
        .sum::<f64>() / all_analyses.len().max(1) as f64;

    println!("\n   📊 Across {} vocalizations:", all_analyses.len());
    println!("      • Total phrases detected: {}", total_phrases);
    println!("      • Total phrase types: {}", total_types);
    println!("      • Average type entropy: {:.3} bits", avg_entropy);

    // Save results
    let results_path = output_dir.join("within_call_analyses.json");
    fs::write(&results_path, serde_json::to_string_pretty(&all_analyses)?)?;
    println!("\n   💾 Results saved to: {}", results_path.display());

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      ANALYSIS COMPLETE                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}
