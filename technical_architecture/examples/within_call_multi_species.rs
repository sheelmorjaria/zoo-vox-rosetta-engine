// =============================================================================
// Within-Call Phrase Discovery: Multi-Species Analysis
// =============================================================================

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// =============================================================================
// Phrase Segmentation
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidate {
    pub id: usize,
    pub start_ms: f64,
    pub end_ms: f64,
    pub duration_ms: f64,
    pub start_sample: usize,
    pub end_sample: usize,
    pub n_samples: usize,
    pub features: Vec<f64>,
    pub phrase_type: Option<i32>,
    pub type_confidence: Option<f64>,
}

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
            min_phrase_ms: 30.0,
            max_phrase_ms: 500.0,
            energy_threshold: 0.05,
            min_gap_ms: 10.0,
            sample_rate: 48000,
        }
    }
}

pub struct PhraseSegmenter {
    config: SegmentationConfig,
}

impl PhraseSegmenter {
    pub fn new(config: SegmentationConfig) -> Self {
        Self { config }
    }

    pub fn segment(&self, audio: &[f32]) -> Vec<PhraseCandidate> {
        let n = audio.len();
        if n == 0 { return vec![]; }

        let sample_rate = self.config.sample_rate as f64;
        let window_samples = (sample_rate * 0.005) as usize;
        if window_samples == 0 { return vec![]; }
        let n_windows = n / window_samples;
        if n_windows == 0 { return vec![]; }

        let mut energy_profile = Vec::with_capacity(n_windows);
        for i in 0..n_windows {
            let start = i * window_samples;
            let end = (start + window_samples).min(n);
            let rms: f32 = audio[start..end].iter().map(|x| x * x).sum::<f32>().sqrt() / (end - start) as f32;
            energy_profile.push(rms);
        }

        let max_energy = energy_profile.iter().cloned().fold(0.0f32, f32::max);
        if max_energy == 0.0 { return vec![]; }
        let threshold = max_energy * self.config.energy_threshold as f32;

        let min_phrase_windows = (self.config.min_phrase_ms / 5.0).max(1.0) as usize;
        let min_gap_windows = (self.config.min_gap_ms / 5.0).max(1.0) as usize;

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
                }
                silence_count = 0;
            } else if in_phrase {
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
                            duration_ms: (end_sample.saturating_sub(start_sample)) as f64 / sample_rate * 1000.0,
                            start_sample,
                            end_sample,
                            n_samples: end_sample.saturating_sub(start_sample),
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
                    duration_ms: (end_sample.saturating_sub(start_sample)) as f64 / sample_rate * 1000.0,
                    start_sample,
                    end_sample,
                    n_samples: end_sample.saturating_sub(start_sample),
                    features: Vec::new(),
                    phrase_type: None,
                    type_confidence: None,
                });
            }
        }

        candidates.into_iter()
            .filter(|c| c.duration_ms <= self.config.max_phrase_ms && c.n_samples > 0)
            .enumerate()
            .map(|(i, mut c)| { c.id = i; c })
            .collect()
    }
}

// =============================================================================
// Feature Extraction (15D)
// =============================================================================

pub struct SimpleFeatureExtractor {
    sample_rate: u32,
}

impl SimpleFeatureExtractor {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    pub fn extract(&self, audio: &[f32]) -> Vec<f64> {
        let n = audio.len();
        if n == 0 { return vec![0.0; 15]; }

        let mut features = vec![0.0f64; 15];

        features[0] = (n as f64 / self.sample_rate as f64 * 1000.0).min(500.0) / 500.0;
        features[1] = (audio.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();

        let zcr = audio.windows(2)
            .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
            .count() as f64 / n as f64;
        features[2] = zcr;
        features[3] = zcr * 2.0;

        let mean: f64 = audio.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
        let var: f64 = audio.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        features[4] = var.sqrt();
        features[5] = mean;

        let envelope_mean: f64 = audio.iter().map(|x| x.abs() as f64).sum::<f64>() / n as f64;
        features[6] = envelope_mean;

        let max_val = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_val > 0.0 {
            let peak_pos = audio.iter().enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            features[7] = peak_pos as f64 / n as f64;

            let threshold = max_val * 0.9;
            let attack_sample = audio.iter().position(|&x| x.abs() >= threshold).unwrap_or(n);
            features[8] = attack_sample as f64 / n as f64;

            let low_thresh = max_val * 0.1;
            let mut decay_end = peak_pos;
            for i in peak_pos..n {
                if audio[i].abs() < low_thresh {
                    decay_end = i;
                    break;
                }
            }
            features[9] = (decay_end.saturating_sub(peak_pos)) as f64 / n as f64;
        }

        let window_size = 32;
        let mut freqs = Vec::new();
        for i in (0..n.saturating_sub(window_size)).step_by(window_size) {
            let window = &audio[i..i + window_size];
            let zc = window.windows(2)
                .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
                .count() as f64 / window_size as f64;
            freqs.push(zc);
        }
        if freqs.len() >= 2 {
            let fmean = freqs.iter().sum::<f64>() / freqs.len() as f64;
            let fvar = freqs.iter().map(|f| (f - fmean).powi(2)).sum::<f64>() / freqs.len() as f64;
            features[10] = fvar.sqrt();
        }

        let mut envelope = Vec::new();
        for i in (0..n.saturating_sub(window_size)).step_by(window_size / 2) {
            let window = &audio[i..i + window_size];
            let rms = (window.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / window_size as f64).sqrt();
            envelope.push(rms);
        }
        if envelope.len() >= 2 {
            let emean = envelope.iter().sum::<f64>() / envelope.len() as f64;
            let evar = envelope.iter().map(|e| (e - emean).powi(2)).sum::<f64>() / envelope.len() as f64;
            features[11] = evar.sqrt() / (emean + 1e-10);
        }

        let lag_range = (n / 4).min(500);
        if lag_range > 1 {
            let centered: Vec<f64> = audio.iter().map(|x| *x as f64 - mean).collect();
            let mut max_ac: f64 = 0.0;
            for lag in 1..lag_range {
                let mut sum = 0.0;
                for i in 0..(n - lag) {
                    sum += centered[i] * centered[i + lag];
                }
                max_ac = max_ac.max(sum);
            }
            let ac0 = centered.iter().map(|x| x * x).sum::<f64>();
            if ac0 > 0.0 { features[12] = max_ac / ac0; }
        }

        if var > 0.0 {
            let skew: f64 = audio.iter()
                .map(|x| ((*x as f64 - mean) / var.sqrt()).powi(3))
                .sum::<f64>() / n as f64;
            features[13] = skew.max(-2.0).min(2.0) / 2.0;

            let kurt: f64 = audio.iter()
                .map(|x| ((*x as f64 - mean) / var.sqrt()).powi(4))
                .sum::<f64>() / n as f64;
            features[14] = (kurt - 3.0).max(-3.0).min(10.0) / 10.0;
        }

        for f in &mut features {
            if !f.is_finite() { *f = 0.0; }
        }

        features
    }
}

// =============================================================================
// Similarity Engine
// =============================================================================

pub struct AcousticSimilarityEngine {
    weights: Array1<f64>,
}

impl AcousticSimilarityEngine {
    pub fn new(feature_dim: usize) -> Self {
        let mut weights = Array1::ones(feature_dim);
        if feature_dim > 0 { weights[0] = 1.5; }
        if feature_dim > 1 { weights[1] = 1.8; }
        if feature_dim > 2 { weights[2] = 1.2; }
        if feature_dim > 3 { weights[3] = 2.0; }
        if feature_dim > 10 { weights[10] = 1.8; }
        if feature_dim > 11 { weights[11] = 1.8; }
        if feature_dim > 12 { weights[12] = 2.0; }
        Self { weights }
    }

    pub fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).zip(self.weights.iter())
            .map(|((x, y), w)| w * (x - y).powi(2))
            .sum::<f64>().sqrt()
    }

    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        1.0 / (1.0 + self.distance(a, b))
    }
}

// =============================================================================
// Within-Call Analysis
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallAnalysis {
    pub file_name: String,
    pub total_duration_ms: f64,
    pub phrases: Vec<PhraseCandidate>,
    pub n_phrase_types: usize,
    pub phrase_types: Vec<i32>,
    pub phrase_sequence: Vec<i32>,
    pub stats: WithinCallStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallStats {
    pub n_phrases: usize,
    pub avg_phrase_duration_ms: f64,
    pub type_distribution: HashMap<i32, usize>,
    pub type_entropy: f64,
    pub avg_within_type_similarity: f64,
    pub phrase_rate: f64,
}

pub struct WithinCallAnalyzer {
    segmenter: PhraseSegmenter,
    same_type_threshold: f64,
    feature_dim: usize,
}

impl WithinCallAnalyzer {
    pub fn new(config: SegmentationConfig, feature_dim: usize) -> Self {
        Self {
            segmenter: PhraseSegmenter::new(config),
            same_type_threshold: 0.70,
            feature_dim,
        }
    }

    pub fn analyze(&self, audio: &[f32], file_name: &str, sample_rate: u32) -> WithinCallAnalysis {
        let mut phrases = self.segmenter.segment(audio);
        let extractor = SimpleFeatureExtractor::new(sample_rate);

        for phrase in &mut phrases {
            let end = phrase.end_sample.min(audio.len());
            let start = phrase.start_sample.min(end);
            if end > start {
                phrase.features = extractor.extract(&audio[start..end]);
            }
        }

        let n = phrases.len();
        let engine = AcousticSimilarityEngine::new(self.feature_dim);

        let mut similarity_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let sim = engine.similarity(&phrases[i].features, &phrases[j].features);
                similarity_matrix[i][j] = sim;
                similarity_matrix[j][i] = sim;
            }
        }

        let (phrase_types, n_phrase_types) = self.discover_phrase_types(n, &similarity_matrix);

        for (phrase, &ptype) in phrases.iter_mut().zip(phrase_types.iter()) {
            phrase.phrase_type = Some(ptype);
        }

        let type_distribution: HashMap<i32, usize> = phrase_types.iter()
            .fold(HashMap::new(), |mut acc, &t| {
                *acc.entry(t).or_default() += 1;
                acc
            });

        let type_entropy = if n > 0 {
            let mut entropy = 0.0;
            for &count in type_distribution.values() {
                let p = count as f64 / n as f64;
                if p > 0.0 { entropy -= p * p.log2(); }
            }
            entropy
        } else { 0.0 };

        let avg_within_type_similarity = self.compute_within_type_similarity(&phrase_types, &similarity_matrix);

        let total_duration_ms: f64 = phrases.iter().map(|p| p.duration_ms).sum();
        let phrase_rate = if total_duration_ms > 0.0 {
            n as f64 / (total_duration_ms / 1000.0)
        } else { 0.0 };

        let stats = WithinCallStats {
            n_phrases: n,
            avg_phrase_duration_ms: if n > 0 {
                phrases.iter().map(|p| p.duration_ms).sum::<f64>() / n as f64
            } else { 0.0 },
            type_distribution,
            type_entropy,
            avg_within_type_similarity,
            phrase_rate,
        };

        WithinCallAnalysis {
            file_name: file_name.to_string(),
            total_duration_ms: audio.len() as f64 / sample_rate as f64 * 1000.0,
            phrases,
            n_phrase_types,
            phrase_types: phrase_types.clone(),
            phrase_sequence: phrase_types,
            stats,
        }
    }

    fn discover_phrase_types(&self, n: usize, similarity_matrix: &[Vec<f64>]) -> (Vec<i32>, usize) {
        if n == 0 { return (vec![], 0); }

        let mut phrase_types = vec![-1i32; n];
        let mut next_type = 0i32;

        for i in 0..n {
            if phrase_types[i] != -1 { continue; }

            let mut best_type = -1i32;
            let mut best_sim = 0.0f64;

            for j in 0..i {
                if phrase_types[j] != -1 {
                    let sim = similarity_matrix[i][j];
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
                    if phrase_types[j] == -1 && similarity_matrix[i][j] >= self.same_type_threshold {
                        phrase_types[j] = next_type;
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

        if count > 0 { total_sim / count as f64 } else { 1.0 }
    }
}

// =============================================================================
// Audio Loading
// =============================================================================

fn load_audio_file(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(ext.to_string_lossy().as_ref());
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;

    let track = format.default_track().ok_or("No default track")?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(48000);

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    let mut samples = Vec::new();
    let mut sample_buf = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id { continue; }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Get channel count before consuming decoded
        let n_channels = decoded.spec().channels.count() as usize;

        if sample_buf.is_none() {
            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;
            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
        }

        if let Some(ref mut buf) = sample_buf {
            buf.copy_interleaved_ref(decoded);
            let buf_samples = buf.samples();

            // Convert to mono by averaging channels
            for chunk in buf_samples.chunks(n_channels.max(1)) {
                let mono: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                samples.push(mono);
            }
        }
    }

    Ok((samples, sample_rate))
}

// =============================================================================
// Dataset Analysis
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub dataset_name: String,
    pub total_files: usize,
    pub total_phrases: usize,
    pub avg_phrases_per_file: f64,
    pub avg_phrase_types_per_file: f64,
    pub avg_phrase_duration_ms: f64,
    pub avg_type_entropy: f64,
    pub avg_within_type_similarity: f64,
    pub file_analyses: Vec<WithinCallAnalysis>,
}

fn find_audio_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_audio_files(&path));
            } else if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ["wav", "flac", "mp3", "ogg"].contains(&ext.as_str()) {
                    if !path.to_string_lossy().contains("__MACOSX") {
                        files.push(path);
                    }
                }
            }
        }
    }
    
    files
}

fn analyze_dataset(
    name: &str,
    audio_dir: &Path,
    max_files: Option<usize>,
    output_path: &Path,
) -> Result<DatasetAnalysis, Box<dyn Error>> {
    println!("\n{}", "=".repeat(70));
    println!("Within-Call Phrase Discovery: {}", name);
    println!("{}", "=".repeat(70));

    let mut audio_files = find_audio_files(audio_dir);
    audio_files.sort();
    
    if let Some(max) = max_files {
        audio_files.truncate(max);
    }

    println!("Found {} audio files", audio_files.len());

    if audio_files.is_empty() {
        return Ok(DatasetAnalysis {
            dataset_name: name.to_string(),
            total_files: 0,
            total_phrases: 0,
            avg_phrases_per_file: 0.0,
            avg_phrase_types_per_file: 0.0,
            avg_phrase_duration_ms: 0.0,
            avg_type_entropy: 0.0,
            avg_within_type_similarity: 0.0,
            file_analyses: vec![],
        });
    }

    let mut file_analyses: Vec<WithinCallAnalysis> = Vec::new();

    for (i, path) in audio_files.iter().enumerate() {
        if i % 100 == 0 {
            println!("  Processing {}/{}: {}", i + 1, audio_files.len(), path.file_name().unwrap().to_string_lossy());
        }

        match load_audio_file(path) {
            Ok((audio, sample_rate)) => {
                if audio.len() < 1000 { continue; }

                let config = SegmentationConfig { sample_rate, ..Default::default() };
                let analyzer = WithinCallAnalyzer::new(config, 15);

                let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                let analysis = analyzer.analyze(&audio, &file_name, sample_rate);

                if analysis.phrases.len() > 0 {
                    file_analyses.push(analysis);
                }
            }
            Err(_) => {}
        }
    }

    println!("  Analyzed {} files", file_analyses.len());

    let total_files = file_analyses.len();
    let total_phrases: usize = file_analyses.iter().map(|a| a.stats.n_phrases).sum();

    let avg_phrases_per_file = if total_files > 0 { total_phrases as f64 / total_files as f64 } else { 0.0 };
    let avg_phrase_types_per_file = if total_files > 0 {
        file_analyses.iter().map(|a| a.n_phrase_types as f64).sum::<f64>() / total_files as f64
    } else { 0.0 };
    let avg_phrase_duration_ms = if total_phrases > 0 {
        file_analyses.iter().flat_map(|a| a.phrases.iter().map(|p| p.duration_ms)).sum::<f64>() / total_phrases as f64
    } else { 0.0 };
    let avg_type_entropy = if total_files > 0 {
        file_analyses.iter().map(|a| a.stats.type_entropy).sum::<f64>() / total_files as f64
    } else { 0.0 };
    let avg_within_type_similarity = if total_files > 0 {
        file_analyses.iter().map(|a| a.stats.avg_within_type_similarity).sum::<f64>() / total_files as f64
    } else { 0.0 };

    let dataset_analysis = DatasetAnalysis {
        dataset_name: name.to_string(),
        total_files,
        total_phrases,
        avg_phrases_per_file,
        avg_phrase_types_per_file,
        avg_phrase_duration_ms,
        avg_type_entropy,
        avg_within_type_similarity,
        file_analyses,
    };

    println!("\n--- Summary: {} ---", name);
    println!("  Files: {}", total_files);
    println!("  Total phrases: {}", total_phrases);
    println!("  Avg phrases/file: {:.2}", avg_phrases_per_file);
    println!("  Avg phrase types/file: {:.2}", avg_phrase_types_per_file);
    println!("  Avg phrase duration: {:.1} ms", avg_phrase_duration_ms);
    println!("  Avg type entropy: {:.3} bits", avg_type_entropy);

    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &dataset_analysis)?;
    println!("Saved to: {}", output_path.display());

    Ok(dataset_analysis)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Within-Call Phrase Discovery: Multi-Species Analysis");

    let base_dir = PathBuf::from("/home/sheel/birdsong_analysis/data");
    let output_dir = PathBuf::from("/home/sheel/birdsong_analysis/within_call_results");
    std::fs::create_dir_all(&output_dir)?;

    let datasets = vec![
        ("bird_songs", base_dir.join("bird_songs"), None),
        ("macaques", base_dir.join("macaques/train"), Some(1000)),
        ("zebra_finch_songs", base_dir.join("zebra_finch_songs"), None),
        ("giant_otter", base_dir.join("giant_otter/giant_otters"), None),
        ("orcas", base_dir.join("orcas"), None),
    ];

    let mut all_results: Vec<DatasetAnalysis> = Vec::new();

    for (name, path, max_files) in datasets {
        if !path.exists() {
            println!("\nSkipping {} - not found: {}", name, path.display());
            continue;
        }

        let output_path = output_dir.join(format!("{}_within_call.json", name));
        match analyze_dataset(name, &path, max_files, &output_path) {
            Ok(analysis) => all_results.push(analysis),
            Err(e) => println!("Error analyzing {}: {}", name, e),
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("CROSS-SPECIES COMPARISON");
    println!("{}", "=".repeat(70));
    println!("{:<18} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Species", "Files", "Phrases", "Phrs/Fl", "Types/Fl", "Entropy");
    println!("{}", "-".repeat(70));

    for a in &all_results {
        println!("{:<18} {:>8} {:>10} {:>10.1} {:>10.1} {:>10.3}",
            a.dataset_name, a.total_files, a.total_phrases,
            a.avg_phrases_per_file, a.avg_phrase_types_per_file, a.avg_type_entropy);
    }

    println!("\nDone!");
    Ok(())
}
