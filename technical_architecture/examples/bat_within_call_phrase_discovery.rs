// Within-Call Phrase Discovery for Egyptian Fruit Bats
//
// Analyzes the internal structure of individual bat vocalizations to discover:
// - Repeated phrase types within a single call
// - Phrase sequences and potential "syntax"
// - Motif patterns (repeated phrase sequences)
// - Individual vocal repertoire
//
// Bat-specific: 250kHz sample rate for ultrasonic vocalizations

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
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

use technical_architecture::{AcousticSimilarityEngine, MicroDynamicsExtractor};

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    pub min_phrase_ms: f64,
    pub max_phrase_ms: f64,
    pub energy_threshold: f64,
    pub min_gap_ms: f64,
    pub fixed_window_ms: Option<f64>,
    pub sample_rate: u32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            min_phrase_ms: 10.0,    // Bats have shorter phrases
            max_phrase_ms: 200.0,   // Max 200ms phrases
            energy_threshold: 0.03, // Lower threshold for bats
            min_gap_ms: 5.0,        // Shorter gaps
            fixed_window_ms: None,
            sample_rate: 250000, // Bat ultrasonic sample rate
        }
    }
}

// =============================================================================
// Phrase Candidate
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidate {
    pub id: usize,
    pub start_ms: f64,
    pub end_ms: f64,
    pub duration_ms: f64,
    pub start_sample: usize,
    pub end_sample: usize,
    pub features: Vec<f64>,
    pub phrase_type: Option<i32>,
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

    pub fn segment(&self, audio: &[f32]) -> Vec<PhraseCandidate> {
        self.segment_adaptive(audio)
    }

    fn segment_adaptive(&self, audio: &[f32]) -> Vec<PhraseCandidate> {
        let n = audio.len();
        let sample_rate = self.config.sample_rate as f64;

        if n == 0 {
            return vec![];
        }

        // Use smaller windows for high-frequency bat audio (2ms windows)
        let window_samples = (sample_rate * 0.002) as usize;
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
            let rms: f32 = audio[start..end].iter().map(|x| x * x).sum::<f32>().sqrt() / (end - start) as f32;
            energy_profile.push(rms);
        }

        let max_energy = energy_profile.iter().cloned().fold(0.0f32, f32::max);
        if max_energy == 0.0 {
            return vec![];
        }
        let threshold = max_energy * self.config.energy_threshold as f32;

        let min_phrase_windows = (self.config.min_phrase_ms / 2.0) as usize;
        let min_gap_windows = (self.config.min_gap_ms / 2.0) as usize;

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

        candidates
            .into_iter()
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

    pub fn analyze(&self, audio: &[f32], file_name: &str, call_type: Option<&str>) -> WithinCallAnalysis {
        let mut phrases = self.segmenter.segment(audio);

        // Extract features for each phrase
        let extractor = MicroDynamicsExtractor::new(self.segmenter.config.sample_rate);
        for phrase in &mut phrases {
            let phrase_audio = &audio[phrase.start_sample..phrase.end_sample];
            if let Ok(features) = extractor.extract(phrase_audio) {
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
                phrase.features.extend(features.mfcc.iter().map(|&v| v as f64));
                phrase.features.push(features.spectral_flux as f64);
                phrase.features.push(features.median_ici_ms as f64);
                phrase.features.push(features.onset_rate_hz as f64);
                phrase.features.push(features.ici_coefficient_of_variation as f64);
            }
        }

        let n = phrases.len();

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

        let (phrase_types, n_phrase_types) = self.discover_phrase_types(&phrases, &distance_matrix);

        for (phrase, &ptype) in phrases.iter_mut().zip(phrase_types.iter()) {
            phrase.phrase_type = Some(ptype);

            let same_type: Vec<usize> = phrase_types
                .iter()
                .enumerate()
                .filter(|(_, &t)| t == ptype)
                .map(|(i, _)| i)
                .collect();

            if same_type.len() > 1 {
                let avg_sim: f64 = same_type
                    .iter()
                    .filter(|&&idx| idx != phrase.id)
                    .map(|&idx| similarity_matrix[phrase.id][idx])
                    .sum::<f64>()
                    / (same_type.len() - 1) as f64;
                phrase.type_confidence = Some(avg_sim);
            } else {
                phrase.type_confidence = Some(1.0);
            }
        }

        let motifs = self.discover_motifs(&phrase_types);
        let transition_matrix = self.build_transition_matrix(&phrase_types);
        let stats = self.compute_stats(
            &phrases,
            &phrase_types,
            n_phrase_types,
            &similarity_matrix,
            &distance_matrix,
        );

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

    fn discover_phrase_types(&self, phrases: &[PhraseCandidate], distance_matrix: &[Vec<f64>]) -> (Vec<i32>, usize) {
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
            let probs: HashMap<i32, f64> = to_counts
                .into_iter()
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

        let avg_within_type_similarity = self.compute_within_type_similarity(phrase_types, similarity_matrix);

        let avg_between_type_distance = self.compute_between_type_distance(phrase_types, distance_matrix);

        let total_duration_ms: f64 = phrases.iter().map(|p| p.duration_ms).sum();
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

    fn compute_within_type_similarity(&self, phrase_types: &[i32], similarity_matrix: &[Vec<f64>]) -> f64 {
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

        if count > 0 {
            total_sim / count as f64
        } else {
            1.0
        }
    }

    fn compute_between_type_distance(&self, phrase_types: &[i32], distance_matrix: &[Vec<f64>]) -> f64 {
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

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }
}

impl WithinCallAnalysis {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║     Bat Within-Call Phrase Analysis Summary                    ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📁 File: {}", self.file_name);
        if let Some(ct) = &self.call_type {
            println!("   Call type: {}", ct);
        }
        println!("   Duration: {:.1} ms", self.total_duration_ms);

        println!("\n📊 Phrase Statistics:");
        println!("   • Total phrases: {}", self.stats.n_phrases);
        println!("   • Unique phrase types: {}", self.n_phrase_types);
        println!(
            "   • Average phrase duration: {:.1} ms",
            self.stats.avg_phrase_duration_ms
        );
        println!("   • Phrase rate: {:.2} phrases/sec", self.stats.phrase_rate);
        println!("   • Type entropy: {:.3} bits", self.stats.type_entropy);

        if !self.motifs.is_empty() {
            println!("\n📊 Discovered Motifs:");
            for motif in self.motifs.iter().take(5) {
                let pattern_str: String = motif
                    .pattern
                    .iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join("-");
                println!("   • Pattern [{}]: {} occurrences", pattern_str, motif.occurrences);
            }
        }
    }
}

// =============================================================================
// WAV Audio Loading (for 250kHz bat recordings)
// =============================================================================

fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("wav");

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
            AudioBufferRef::S32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i32::MAX as f32));
                }
            }
            AudioBufferRef::F64(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32));
                }
            }
            _ => {}
        }
    }

    Ok(audio_samples)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Bat Within-Call Phrase Discovery (250kHz Ultrasonic)        ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  🦇 Analyzes INTERNAL structure of bat vocalizations           ║");
    println!("║  📊 Discovers repeated phrases, motifs, and \"syntax\"           ║");
    println!("║  🎯 Uses SIMILARITY THRESHOLDING (not clustering)              ║");
    println!("║  🔊 Optimized for 250kHz ultrasonic recordings                 ║");
    println!("║                                                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Bat dataset configuration
    let vocalizations_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");
    let output_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/within_call_phrase_results");

    if !vocalizations_dir.exists() {
        println!("❌ Bat audio directory not found: {}", vocalizations_dir.display());
        return Ok(());
    }

    fs::create_dir_all(&output_dir)?;

    // Find WAV files
    let mut wav_files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&vocalizations_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "wav" {
                    wav_files.push(path);
                }
            }
        }
    }

    let total_files = wav_files.len();
    println!("📂 Found {} WAV files to process", total_files);
    println!();

    if total_files == 0 {
        println!("❌ No WAV files found!");
        return Ok(());
    }

    // Configure analyzer for bats (250kHz)
    let config = SegmentationConfig {
        min_phrase_ms: 10.0,    // Bats have shorter syllables
        max_phrase_ms: 200.0,   // Max 200ms phrases
        energy_threshold: 0.03, // Lower threshold for bats
        min_gap_ms: 5.0,        // Shorter gaps between phrases
        sample_rate: 250000,    // 250kHz for ultrasonic
        ..Default::default()
    };

    // Progress tracking
    let processed = AtomicUsize::new(0);
    let errors_count = AtomicUsize::new(0);
    let phrases_count = AtomicUsize::new(0);
    let types_count = AtomicUsize::new(0);
    let start_time = Instant::now();

    let checkpoint_interval = if total_files > 10000 { 10000 } else { 1000 };
    let all_analyses: Mutex<Vec<WithinCallAnalysis>> = Mutex::new(Vec::new());

    println!(
        "🚀 Starting PARALLEL analysis with {} threads...",
        rayon::current_num_threads()
    );
    println!("   Total files: {}", total_files);
    println!("   Sample rate: 250kHz (ultrasonic)");
    println!("   Checkpoints every {} files", checkpoint_interval);
    println!();

    // Process in batches
    let batch_size = 1000;
    let num_batches = (total_files + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(total_files);
        let batch: Vec<_> = wav_files[start..end].to_vec();

        let results: Vec<Option<WithinCallAnalysis>> = batch
            .par_iter()
            .map(|path| {
                let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
                let analyzer = WithinCallAnalyzer::new(config.clone(), 30);

                match load_wav_file(path) {
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

        println!(
            "   [{}/{}] {:.1}% | {:.0} files/min | ETA: {:.0}s | Phrases: {} | Errors: {}",
            processed_val,
            total_files,
            processed_val as f64 / total_files as f64 * 100.0,
            rate * 60.0,
            remaining,
            phrases,
            errors
        );

        // Checkpoint save
        if (batch_idx + 1) * batch_size >= checkpoint_interval
            && ((batch_idx + 1) * batch_size) % checkpoint_interval == 0
        {
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
    let avg_entropy: f64 =
        all_analyses.iter().map(|a| a.stats.type_entropy).sum::<f64>() / all_analyses.len().max(1) as f64;

    let files_with_motifs: usize = all_analyses.iter().filter(|a| !a.motifs.is_empty()).count();

    println!("\n   📊 Across {} bat vocalizations:", all_analyses.len());
    println!("      • Total phrases detected: {}", total_phrases);
    println!("      • Total phrase types: {}", total_types);
    println!("      • Average type entropy: {:.3} bits", avg_entropy);
    println!(
        "      • Files with motifs: {} ({:.1}%)",
        files_with_motifs,
        files_with_motifs as f64 / all_analyses.len().max(1) as f64 * 100.0
    );

    // Save results
    let results_path = output_dir.join("bat_within_call_analyses.json");
    fs::write(&results_path, serde_json::to_string_pretty(&all_analyses)?)?;
    println!("\n   💾 Results saved to: {}", results_path.display());

    // Save summary statistics
    let summary = serde_json::json!({
        "total_files": all_analyses.len(),
        "total_phrases": total_phrases,
        "total_phrase_types": total_types,
        "average_entropy": avg_entropy,
        "files_with_motifs": files_with_motifs,
        "errors": errors_count.load(Ordering::Relaxed),
    });
    let summary_path = output_dir.join("summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    println!("   💾 Summary saved to: {}", summary_path.display());

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      ANALYSIS COMPLETE                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}
