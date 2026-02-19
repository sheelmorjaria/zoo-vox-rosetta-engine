//! Within-Call Phrase Discovery for Marmoset Vocalizations
//!
//! Analyzes the internal structure of individual marmoset vocalizations to discover:
//! - Repeated phrase types within a single call
//! - Phrase sequences and potential "syntax"
//! - Motif patterns (repeated phrase sequences)
//! - Within-call vocabulary complexity
//!
//! Dataset: ~/birdsong_analysis/data/Vocalizations (FLAC files in subdirectories)
//! Sample rate: 96kHz (marmoset recordings)

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    pub min_phrase_ms: f64,
    pub max_phrase_ms: f64,
    pub energy_threshold: f64,
    pub min_gap_ms: f64,
    pub sample_rate: u32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            min_phrase_ms: 30.0,    // Marmoset syllables
            max_phrase_ms: 500.0,   // Long phee phrases
            energy_threshold: 0.05, // 5% of max energy
            min_gap_ms: 15.0,       // 15ms minimum gap
            sample_rate: 96000,     // Marmoset recordings
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
        let n = audio.len();
        if n == 0 {
            return vec![];
        }

        let sample_rate = self.config.sample_rate as f64;

        // Compute RMS energy in windows (5ms)
        let window_samples = (sample_rate * 0.005) as usize;
        let n_windows = n / window_samples;

        if n_windows == 0 {
            return vec![];
        }

        let mut energy_profile = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            let start = i * window_samples;
            let end = (start + window_samples).min(n);
            let rms: f32 =
                audio[start..end].iter().map(|x| x * x).sum::<f32>().sqrt() / (end - start) as f32;
            energy_profile.push(rms);
        }

        let max_energy = energy_profile.iter().cloned().fold(0.0f32, f32::max);
        if max_energy == 0.0 {
            return vec![];
        }
        let threshold = max_energy * self.config.energy_threshold as f32;

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
            } else if in_phrase {
                silence_count += 1;
                if silence_count >= min_gap_windows {
                    let phrase_end = i - silence_count;
                    let phrase_len = phrase_end - phrase_start;

                    if phrase_len >= min_phrase_windows {
                        let start_sample = phrase_start * window_samples;
                        let end_sample = phrase_end * window_samples;

                        let duration_ms = (end_sample - start_sample) as f64 / sample_rate * 1000.0;
                        if duration_ms <= self.config.max_phrase_ms {
                            candidates.push(PhraseCandidate {
                                id: phrase_count,
                                start_ms: start_sample as f64 / sample_rate * 1000.0,
                                end_ms: end_sample as f64 / sample_rate * 1000.0,
                                duration_ms,
                                start_sample,
                                end_sample,
                                features: Vec::new(),
                                phrase_type: None,
                                type_confidence: None,
                            });
                            phrase_count += 1;
                        }
                    }
                    in_phrase = false;
                }
            }
        }

        // Handle final phrase
        if in_phrase {
            let phrase_end = energy_profile.len() - silence_count;
            let phrase_len = phrase_end - phrase_start;

            if phrase_len >= min_phrase_windows {
                let start_sample = phrase_start * window_samples;
                let end_sample = (phrase_end * window_samples).min(n);
                let duration_ms = (end_sample - start_sample) as f64 / sample_rate * 1000.0;

                if duration_ms <= self.config.max_phrase_ms {
                    candidates.push(PhraseCandidate {
                        id: phrase_count,
                        start_ms: start_sample as f64 / sample_rate * 1000.0,
                        end_ms: end_sample as f64 / sample_rate * 1000.0,
                        duration_ms,
                        start_sample,
                        end_sample,
                        features: Vec::new(),
                        phrase_type: None,
                        type_confidence: None,
                    });
                }
            }
        }

        candidates
    }
}

// =============================================================================
// Feature Extraction (30D)
// =============================================================================

pub fn extract_30d_features(audio: &[f32], sample_rate: u32) -> Vec<f64> {
    let n = audio.len();
    if n == 0 {
        return vec![0.0; 30];
    }

    let sr = sample_rate as f64;
    let mut features = vec![0.0; 30];

    // 0: Duration (ms)
    features[0] = n as f64 / sr * 1000.0;

    // 1-4: Energy statistics
    let mean_energy: f64 = audio.iter().map(|x| (x * x) as f64).sum::<f64>() / n as f64;
    features[1] = mean_energy.sqrt(); // RMS

    let max_amp = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max) as f64;
    features[2] = max_amp;

    let energy_var: f64 = audio
        .iter()
        .map(|x| {
            let e = (x * x) as f64;
            (e - mean_energy).powi(2)
        })
        .sum::<f64>()
        / n as f64;
    features[3] = energy_var.sqrt();

    // 5-6: Zero crossing rate
    let zcr: f64 = audio
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count() as f64
        / n as f64;
    features[5] = zcr;
    features[6] = zcr * sr / 2.0;

    // 7-9: Spectral centroid approximation
    let window_size = 1024.min(n / 4).max(64);
    if n >= window_size * 2 {
        let mut centroid_sum = 0.0;
        let mut centroid_count = 0;

        for start in (0..n - window_size).step_by(window_size / 2) {
            let end = (start + window_size).min(n);
            let window = &audio[start..end];

            let mut magnitudes = vec![0.0f64; window_size / 2];
            for k in 0..window_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (i, &s) in window.iter().enumerate() {
                    let angle =
                        2.0 * std::f64::consts::PI * k as f64 * i as f64 / window_size as f64;
                    real += s as f64 * angle.cos();
                    imag += s as f64 * angle.sin();
                }
                magnitudes[k] = (real * real + imag * imag).sqrt();
            }

            let total_mag: f64 = magnitudes.iter().sum();
            if total_mag > 0.0 {
                let centroid: f64 = magnitudes
                    .iter()
                    .enumerate()
                    .map(|(k, &m)| k as f64 * m)
                    .sum::<f64>()
                    / total_mag;
                centroid_sum += centroid * sr / window_size as f64;
                centroid_count += 1;
            }
        }

        if centroid_count > 0 {
            features[7] = centroid_sum / centroid_count as f64;
        }
    }

    // 10-14: Envelope features
    let envelope: Vec<f64> = audio
        .chunks(100)
        .map(|chunk| chunk.iter().map(|x| x.abs() as f64).sum::<f64>() / chunk.len() as f64)
        .collect();

    if envelope.len() > 1 {
        let max_env = envelope.iter().cloned().fold(0.0f64, f64::max);
        if max_env > 0.0 {
            let threshold_10 = max_env * 0.1;
            let threshold_90 = max_env * 0.9;

            let mut attack_start = 0;
            let mut attack_end = envelope.len();

            for (i, &e) in envelope.iter().enumerate() {
                if e >= threshold_10 && attack_start == 0 {
                    attack_start = i;
                }
                if e >= threshold_90 {
                    attack_end = i;
                    break;
                }
            }

            let attack_samples = (attack_end - attack_start) * 100;
            features[10] = attack_samples as f64 / sr * 1000.0;
        }
    }

    // 15-19: Amplitude modulation
    if n > 200 {
        let mod_period = (sr / 50.0) as usize;
        let n_mod_windows = n / mod_period;

        if n_mod_windows > 2 {
            let mut mod_energies = Vec::with_capacity(n_mod_windows);
            for i in 0..n_mod_windows {
                let start = i * mod_period;
                let end = (start + mod_period).min(n);
                let energy: f64 = audio[start..end]
                    .iter()
                    .map(|x| (x * x) as f64)
                    .sum::<f64>()
                    / (end - start) as f64;
                mod_energies.push(energy.sqrt());
            }

            let mod_mean: f64 = mod_energies.iter().sum::<f64>() / mod_energies.len() as f64;
            if mod_mean > 0.0 {
                let mod_var: f64 = mod_energies
                    .iter()
                    .map(|e| (e - mod_mean).powi(2))
                    .sum::<f64>()
                    / mod_energies.len() as f64;
                features[15] = mod_var.sqrt() / mod_mean;
            }
        }
    }

    // 20-24: Fundamental frequency (autocorrelation)
    if n > 256 {
        let min_period = (sr / 20000.0) as usize;
        let max_period = (sr / 500.0).min(n as f64 / 2.0) as usize;

        if max_period > min_period {
            let mean: f32 = audio.iter().sum::<f32>() / n as f32;
            let centered: Vec<f32> = audio.iter().map(|x| x - mean).collect();

            let mut best_period = min_period;
            let mut best_corr = 0.0f32;

            for period in min_period..max_period {
                let mut corr = 0.0f32;
                for i in 0..(n - period) {
                    corr += centered[i] * centered[i + period];
                }
                if corr > best_corr {
                    best_corr = corr;
                    best_period = period;
                }
            }

            if best_corr > 0.0 {
                features[20] = sr / best_period as f64;
            }
        }
    }

    features
}

// =============================================================================
// Within-Call Analysis Results
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallAnalysis {
    pub file_name: String,
    pub total_duration_ms: f64,
    pub phrases: Vec<PhraseCandidate>,
    pub n_phrase_types: usize,
    pub phrase_types: Vec<i32>,
    pub motifs: Vec<Motif>,
    pub phrase_sequence: Vec<i32>,
    pub stats: WithinCallStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    pub pattern: Vec<i32>,
    pub occurrences: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallStats {
    pub n_phrases: usize,
    pub avg_phrase_duration_ms: f64,
    pub type_distribution: HashMap<i32, usize>,
    pub type_entropy: f64,
    pub phrase_rate: f64,
}

// =============================================================================
// Similarity and Phrase Type Discovery
// =============================================================================

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        (dot / (mag_a * mag_b)).max(0.0).min(1.0)
    } else {
        0.0
    }
}

fn discover_phrase_types(phrases: &[PhraseCandidate], threshold: f64) -> (Vec<i32>, usize) {
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
        let mut best_sim = 0.0;

        for j in 0..i {
            if phrase_types[j] != -1 {
                let sim = cosine_similarity(&phrases[i].features, &phrases[j].features);
                if sim >= threshold && sim > best_sim {
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
                    let sim = cosine_similarity(&phrases[i].features, &phrases[j].features);
                    if sim >= threshold {
                        phrase_types[j] = next_type;
                    }
                }
            }

            next_type += 1;
        }
    }

    // Renumber consecutively
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

fn discover_motifs(phrase_types: &[i32]) -> Vec<Motif> {
    let n = phrase_types.len();
    if n < 3 {
        return vec![];
    }

    let mut motif_counts: HashMap<Vec<i32>, usize> = HashMap::new();

    for motif_len in 2..=5.min(n / 2) {
        for i in 0..=(n - motif_len) {
            let pattern: Vec<i32> = phrase_types[i..i + motif_len].to_vec();
            *motif_counts.entry(pattern).or_default() += 1;
        }
    }

    let mut motifs: Vec<Motif> = motif_counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(pattern, occurrences)| Motif {
            pattern,
            occurrences,
        })
        .collect();

    motifs.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
    motifs
}

fn analyze_vocalization(audio: &[f32], file_name: String, sample_rate: u32) -> WithinCallAnalysis {
    let config = SegmentationConfig {
        sample_rate,
        ..Default::default()
    };
    let segmenter = PhraseSegmenter::new(config);

    let mut phrases = segmenter.segment(audio);

    for phrase in &mut phrases {
        let phrase_audio = &audio[phrase.start_sample..phrase.end_sample];
        phrase.features = extract_30d_features(phrase_audio, sample_rate);
    }

    let n = phrases.len();
    let (phrase_types, n_phrase_types) = discover_phrase_types(&phrases, 0.75);

    for (phrase, &ptype) in phrases.iter_mut().zip(phrase_types.iter()) {
        phrase.phrase_type = Some(ptype);
    }

    let motifs = discover_motifs(&phrase_types);

    let mut type_distribution: HashMap<i32, usize> = HashMap::new();
    for &t in &phrase_types {
        *type_distribution.entry(t).or_default() += 1;
    }

    let type_entropy = if n > 0 {
        let mut entropy = 0.0;
        for &count in type_distribution.values() {
            let p = count as f64 / n as f64;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    } else {
        0.0
    };

    let total_duration_ms = audio.len() as f64 / sample_rate as f64 * 1000.0;
    let phrase_rate = if total_duration_ms > 0.0 {
        n as f64 / (total_duration_ms / 1000.0)
    } else {
        0.0
    };

    let stats = WithinCallStats {
        n_phrases: n,
        avg_phrase_duration_ms: if n > 0 {
            phrases.iter().map(|p| p.duration_ms).sum::<f64>() / n as f64
        } else {
            0.0
        },
        type_distribution,
        type_entropy,
        phrase_rate,
    };

    WithinCallAnalysis {
        file_name,
        total_duration_ms,
        phrases,
        n_phrase_types,
        phrase_types: phrase_types.clone(),
        motifs,
        phrase_sequence: phrase_types,
        stats,
    }
}

// =============================================================================
// Audio Loading (FLAC)
// =============================================================================

fn load_flac_file(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
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

    let sample_rate = track.codec_params.sample_rate.unwrap_or(96000);

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
                    audio_samples.extend(samples.iter().map(|s| s.into_i32() as f32 / 8388608.0));
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
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|s| (s.into_u32() as f32 - 8388608.0) / 8388608.0),
                    );
                }
            }
            AudioBufferRef::U32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| (s as f32 - 2147483648.0) / 2147483648.0),
                    );
                }
            }
            AudioBufferRef::S8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / 128.0));
                }
            }
        }
    }

    Ok((audio_samples, sample_rate))
}

/// Recursively find all FLAC files
fn find_flac_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_flac_files(&path));
            } else if path.extension().map(|e| e == "flac").unwrap_or(false) {
                files.push(path);
            }
        }
    }

    files
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   Marmoset Within-Call Phrase Discovery (96kHz)               ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                 ║");
    println!("║  🐵 Analyzes INTERNAL structure of marmoset vocalizations     ║");
    println!("║  📊 Discovers repeated phrases, motifs, and \"syntax\"           ║");
    println!("║  🎯 Uses SIMILARITY THRESHOLDING (not clustering)              ║");
    println!("║  🔊 Optimized for 96kHz marmoset recordings                    ║");
    println!("║                                                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let home = env::var("HOME").expect("HOME not set");
    let data_dir = PathBuf::from(&home).join("birdsong_analysis/data/Vocalizations");

    if !data_dir.exists() {
        eprintln!("Data directory not found: {:?}", data_dir);
        return Ok(());
    }

    println!("Searching for FLAC files in {:?}...", data_dir);
    let files = find_flac_files(&data_dir);
    println!("Found {} FLAC files", files.len());

    if files.is_empty() {
        eprintln!("No FLAC files found!");
        return Ok(());
    }

    let output_dir =
        PathBuf::from(&home).join("birdsong_analysis/data/marmoset_within_call_results");
    fs::create_dir_all(&output_dir)?;

    let start_time = Instant::now();
    let total_files = files.len();

    let results: Mutex<Vec<WithinCallAnalysis>> = Mutex::new(Vec::new());
    let errors: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    let processed = AtomicUsize::new(0);

    println!(
        "\nProcessing {} files with parallel extraction...",
        total_files
    );

    files.par_iter().for_each(|path| {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        match load_flac_file(path) {
            Ok((audio, sample_rate)) => {
                if audio.is_empty() {
                    let mut errs = errors.lock().unwrap();
                    errs.push((file_name, "Empty audio".to_string()));
                    return;
                }

                let analysis = analyze_vocalization(&audio, file_name, sample_rate);

                let mut res = results.lock().unwrap();
                res.push(analysis);

                let count = processed.fetch_add(1, Ordering::SeqCst) + 1;
                if count % 50000 == 0 {
                    println!("  Processed {}/{} files...", count, total_files);
                }
            }
            Err(e) => {
                let mut errs = errors.lock().unwrap();
                errs.push((file_name, format!("{:?}", e)));
            }
        }
    });

    let elapsed = start_time.elapsed();
    let results = results.lock().unwrap();
    let errors = errors.lock().unwrap();
    let processed_count = processed.load(Ordering::SeqCst);

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    PROCESSING COMPLETE                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Files processed: {}", processed_count);
    println!("Errors: {}", errors.len());
    println!("Processing time: {:.2?}", elapsed);
    println!(
        "Rate: {:.0} files/min",
        processed_count as f64 / elapsed.as_secs_f64() * 60.0
    );

    // Aggregate statistics
    let mut total_phrases = 0;
    let mut total_phrase_types = 0;
    let mut all_durations = Vec::new();
    let mut all_entropies = Vec::new();
    let mut files_with_motifs = 0;

    for a in results.iter() {
        total_phrases += a.stats.n_phrases;
        total_phrase_types += a.n_phrase_types;
        all_durations.extend(a.phrases.iter().map(|p| p.duration_ms));
        all_entropies.push(a.stats.type_entropy);
        if !a.motifs.is_empty() {
            files_with_motifs += 1;
        }
    }

    println!("\n📊 AGGREGATE STATISTICS:");
    println!("   Total phrases detected: {}", total_phrases);
    println!("   Total phrase types: {}", total_phrase_types);
    println!(
        "   Avg phrases per call: {:.2}",
        total_phrases as f64 / processed_count as f64
    );
    println!(
        "   Avg phrase types per call: {:.2}",
        total_phrase_types as f64 / processed_count as f64
    );
    println!(
        "   Files with motifs: {} ({:.1}%)",
        files_with_motifs,
        files_with_motifs as f64 / processed_count as f64 * 100.0
    );

    if !all_durations.is_empty() {
        let avg_dur: f64 = all_durations.iter().sum::<f64>() / all_durations.len() as f64;
        let min_dur = all_durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_dur = all_durations.iter().cloned().fold(0.0, f64::max);
        println!(
            "   Phrase duration: min={:.1}ms, max={:.1}ms, avg={:.1}ms",
            min_dur, max_dur, avg_dur
        );
    }

    if !all_entropies.is_empty() {
        let avg_entropy: f64 = all_entropies.iter().sum::<f64>() / all_entropies.len() as f64;
        let low_ent = all_entropies.iter().filter(|&&e| e < 0.5).count();
        let med_ent = all_entropies
            .iter()
            .filter(|&&e| e >= 0.5 && e < 1.5)
            .count();
        let high_ent = all_entropies.iter().filter(|&&e| e >= 1.5).count();
        println!(
            "   Type entropy: avg={:.3}, low(<0.5)={}, med(0.5-1.5)={}, high(>=1.5)={}",
            avg_entropy, low_ent, med_ent, high_ent
        );
    }

    // Save results
    let output_path = output_dir.join("marmoset_within_call_analyses.json");
    let file = File::create(&output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &*results)?;
    println!("\nResults saved to: {:?}", output_path);

    // Save summary
    let summary = serde_json::json!({
        "total_files": total_files,
        "processed_files": processed_count,
        "errors": errors.len(),
        "processing_time_sec": elapsed.as_secs_f64(),
        "files_per_minute": processed_count as f64 / elapsed.as_secs_f64() * 60.0,
        "total_phrases": total_phrases,
        "total_phrase_types": total_phrase_types,
        "files_with_motifs": files_with_motifs,
        "pct_with_motifs": files_with_motifs as f64 / processed_count as f64 * 100.0,
        "avg_phrases_per_call": total_phrases as f64 / processed_count as f64,
        "avg_phrase_types_per_call": total_phrase_types as f64 / processed_count as f64,
    });

    let summary_path = output_dir.join("marmoset_within_call_summary.json");
    let file = File::create(&summary_path)?;
    serde_json::to_writer_pretty(file, &summary)?;
    println!("Summary saved to: {:?}", summary_path);

    if !errors.is_empty() {
        println!("\n⚠️ Errors (first 5):");
        for (file, err) in errors.iter().take(5) {
            println!("   {}: {}", file, err);
        }
    }

    println!("\n✅ Done!");
    Ok(())
}
