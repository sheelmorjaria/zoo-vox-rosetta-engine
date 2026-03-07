// Within-Call Phrase Discovery for Whistle Signals Dataset
//
// Analyzes dolphin/whale whistle recordings to discover internal
// phrase-level structure using acoustic similarity-based clustering.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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
            min_phrase_ms: 20.0,
            max_phrase_ms: 300.0,
            energy_threshold: 0.05,
            min_gap_ms: 10.0,
            sample_rate: 192000,
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
        if n == 0 {
            return vec![];
        }

        let sample_rate = self.config.sample_rate as f64;
        let window_samples = (sample_rate * 0.002) as usize; // 2ms windows for high sample rate
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

        let min_phrase_windows = (self.config.min_phrase_ms / 2.0).max(1.0) as usize;
        let min_gap_windows = (self.config.min_gap_ms / 2.0).max(1.0) as usize;

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

        candidates
            .into_iter()
            .filter(|c| c.duration_ms <= self.config.max_phrase_ms && c.n_samples > 0)
            .enumerate()
            .map(|(i, mut c)| {
                c.id = i;
                c
            })
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
        if n == 0 {
            return vec![0.0; 15];
        }

        let mut features = vec![0.0f64; 15];

        // Duration (normalized to 500ms max)
        features[0] = (n as f64 / self.sample_rate as f64 * 1000.0).min(500.0) / 500.0;

        // RMS Energy
        features[1] = (audio.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();

        // Zero Crossing Rate
        let zcr = audio
            .windows(2)
            .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
            .count() as f64
            / n as f64;
        features[2] = zcr;
        features[3] = zcr * 2.0;

        // Mean and variance
        let mean: f64 = audio.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
        let var: f64 = audio.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        features[4] = var.sqrt();
        features[5] = mean;

        // Envelope mean
        let envelope_mean: f64 = audio.iter().map(|x| x.abs() as f64).sum::<f64>() / n as f64;
        features[6] = envelope_mean;

        let max_val = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_val > 0.0 {
            // Peak position (normalized)
            let peak_pos = audio
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            features[7] = peak_pos as f64 / n as f64;

            // Attack time (normalized)
            let threshold = max_val * 0.9;
            let attack_sample = audio.iter().position(|&x| x.abs() >= threshold).unwrap_or(n);
            features[8] = attack_sample as f64 / n as f64;

            // Decay time (normalized)
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

        // Frequency variation (using ZCR as proxy)
        let window_size = 64;
        let mut freqs = Vec::new();
        for i in (0..n.saturating_sub(window_size)).step_by(window_size) {
            let window = &audio[i..i + window_size];
            let zc = window
                .windows(2)
                .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
                .count() as f64
                / window_size as f64;
            freqs.push(zc);
        }
        if freqs.len() >= 2 {
            let fmean = freqs.iter().sum::<f64>() / freqs.len() as f64;
            let fvar = freqs.iter().map(|f| (f - fmean).powi(2)).sum::<f64>() / freqs.len() as f64;
            features[10] = fvar.sqrt();
        }

        // Envelope dynamics
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

        // Center of mass
        let total_abs: f64 = audio.iter().map(|x| x.abs() as f64).sum();
        if total_abs > 0.0 {
            let com: f64 = audio
                .iter()
                .enumerate()
                .map(|(i, x)| i as f64 * x.abs() as f64)
                .sum::<f64>()
                / total_abs;
            features[12] = com / n as f64;
        }

        // Skewness
        let std_dev = features[4];
        if std_dev > 1e-10 {
            let skewness: f64 = audio
                .iter()
                .map(|x| ((*x as f64 - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n as f64;
            features[13] = skewness;
        }

        // Kurtosis
        if std_dev > 1e-10 {
            let kurtosis: f64 = audio
                .iter()
                .map(|x| ((*x as f64 - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n as f64
                - 3.0;
            features[14] = kurtosis;
        }

        features
    }
}

// =============================================================================
// Acoustic Similarity Clustering
// =============================================================================

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a > 1e-10 && norm_b > 1e-10 {
        (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
    } else {
        0.0
    }
}

pub struct SimilarityClusterer {
    threshold: f64,
}

impl SimilarityClusterer {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn assign_types(&self, phrases: &mut [PhraseCandidate]) {
        let mut next_type_id = 0i32;
        let mut type_representatives: Vec<(i32, Vec<f64>)> = Vec::new();

        for phrase in phrases.iter_mut() {
            let mut best_type: Option<(i32, f64)> = None;

            for (type_id, ref_features) in &type_representatives {
                let sim = cosine_similarity(&phrase.features, ref_features);
                if sim >= self.threshold {
                    if let Some((_, best_sim)) = best_type {
                        if sim > best_sim {
                            best_type = Some((*type_id, sim));
                        }
                    } else {
                        best_type = Some((*type_id, sim));
                    }
                }
            }

            if let Some((type_id, confidence)) = best_type {
                phrase.phrase_type = Some(type_id);
                phrase.type_confidence = Some(confidence);
            } else {
                let new_type = next_type_id;
                type_representatives.push((new_type, phrase.features.clone()));
                phrase.phrase_type = Some(new_type);
                phrase.type_confidence = Some(1.0);
                next_type_id += 1;
            }
        }
    }
}

// =============================================================================
// Audio Loading
// =============================================================================

pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
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
    let sample_rate = track.codec_params.sample_rate.ok_or("No sample rate")?;

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    let mut samples = Vec::new();
    let mut decode_buf = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                let n_channels = spec.channels.count() as usize;

                if decode_buf.is_none() {
                    decode_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, *spec));
                }

                if let Some(ref mut buf) = decode_buf {
                    buf.copy_interleaved_ref(decoded);
                    let buf_samples = buf.samples();

                    // Convert to mono
                    for chunk in buf_samples.chunks(n_channels.max(1)) {
                        let mono: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                        samples.push(mono);
                    }
                }
            }
            Err(_) => break,
        }
    }

    Ok((samples, sample_rate))
}

// =============================================================================
// Results Structures
// =============================================================================

#[derive(Debug, Serialize)]
pub struct FileAnalysis {
    pub file_name: String,
    pub session: String,
    pub phrases: Vec<PhraseCandidate>,
    pub n_phrase_types: usize,
    pub phrase_types: Vec<i32>,
    pub phrase_sequence: Vec<i32>,
    pub stats: FileStats,
}

#[derive(Debug, Serialize)]
pub struct FileStats {
    pub n_phrases: usize,
    pub total_duration_ms: f64,
    pub avg_phrase_duration_ms: f64,
    pub type_entropy: f64,
}

#[derive(Debug, Serialize)]
pub struct DatasetResults {
    pub dataset: String,
    pub total_files: usize,
    pub files_analyzed: usize,
    pub total_phrases: usize,
    pub unique_phrase_types: usize,
    pub file_analyses: Vec<FileAnalysis>,
    pub summary: String,
}

// =============================================================================
// Main Analysis
// =============================================================================

fn analyze_file(
    path: &Path,
    config: SegmentationConfig,
    similarity_threshold: f64,
) -> Result<FileAnalysis, Box<dyn Error>> {
    let file_name = path.file_name().unwrap().to_string_lossy().to_string();
    let session = path
        .parent()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let (audio, sample_rate) = load_audio(path)?;

    if audio.is_empty() {
        return Ok(FileAnalysis {
            file_name,
            session,
            phrases: Vec::new(),
            n_phrase_types: 0,
            phrase_types: Vec::new(),
            phrase_sequence: Vec::new(),
            stats: FileStats {
                n_phrases: 0,
                total_duration_ms: 0.0,
                avg_phrase_duration_ms: 0.0,
                type_entropy: 0.0,
            },
        });
    }

    let mut config = config;
    config.sample_rate = sample_rate;

    let segmenter = PhraseSegmenter::new(config);
    let mut phrases = segmenter.segment(&audio);

    let extractor = SimpleFeatureExtractor::new(sample_rate);
    for phrase in &mut phrases {
        let start = phrase.start_sample;
        let end = phrase.end_sample.min(audio.len());
        phrase.features = extractor.extract(&audio[start..end]);
    }

    let clusterer = SimilarityClusterer::new(similarity_threshold);
    clusterer.assign_types(&mut phrases);

    // Build sequence and stats
    let phrase_sequence: Vec<i32> = phrases.iter().filter_map(|p| p.phrase_type).collect();

    let mut type_counts: HashMap<i32, usize> = HashMap::new();
    for phrase in &phrases {
        if let Some(t) = phrase.phrase_type {
            *type_counts.entry(t).or_default() += 1;
        }
    }

    let phrase_types: Vec<i32> = type_counts.keys().cloned().collect();
    let total_duration_ms: f64 = phrases.iter().map(|p| p.duration_ms).sum();
    let n_phrases = phrases.len();

    let type_entropy = if !type_counts.is_empty() {
        let total: usize = type_counts.values().sum();
        type_counts
            .values()
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.log2()
            })
            .sum()
    } else {
        0.0
    };

    Ok(FileAnalysis {
        file_name,
        session,
        phrases,
        n_phrase_types: phrase_types.len(),
        phrase_types,
        phrase_sequence,
        stats: FileStats {
            n_phrases,
            total_duration_ms,
            avg_phrase_duration_ms: if n_phrases > 0 {
                total_duration_ms / n_phrases as f64
            } else {
                0.0
            },
            type_entropy,
        },
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Within-Call Phrase Discovery: Whistle Signals Dataset                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/home/sheel/birdsong_analysis/data/Whistle_Signals");
    let output_dir = Path::new("/home/sheel/birdsong_analysis/within_call_results");

    std::fs::create_dir_all(output_dir)?;

    // Find all WAV files
    let mut wav_files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            for sub_entry in std::fs::read_dir(&path)? {
                let sub_entry = sub_entry?;
                let sub_path = sub_entry.path();
                if sub_path.extension().map(|ext| ext == "wav").unwrap_or(false) {
                    wav_files.push(sub_path);
                }
            }
        }
    }

    println!("Found {} WAV files", wav_files.len());
    println!();

    // Whistle analysis parameters
    let config = SegmentationConfig {
        min_phrase_ms: 20.0,    // 20ms minimum for whistle segments
        max_phrase_ms: 300.0,   // 300ms maximum
        energy_threshold: 0.05, // 5% threshold
        min_gap_ms: 10.0,       // 10ms minimum gap
        sample_rate: 192000,    // Will be overwritten by actual rate
    };
    let similarity_threshold = 0.70;

    let mut file_analyses = Vec::new();
    let mut total_phrases = 0;
    let mut all_types: HashSet<i32> = HashSet::new();
    let mut errors = 0;

    for (i, path) in wav_files.iter().enumerate() {
        let file_name = path.file_name().unwrap().to_string_lossy();
        print!("\r[{}/{}] Analyzing: {:.50}...", i + 1, wav_files.len(), file_name);
        std::io::stdout().flush()?;

        match analyze_file(path, config.clone(), similarity_threshold) {
            Ok(analysis) => {
                total_phrases += analysis.stats.n_phrases;
                for &t in &analysis.phrase_types {
                    all_types.insert(t);
                }
                file_analyses.push(analysis);
            }
            Err(e) => {
                errors += 1;
                if errors <= 5 {
                    println!("\n  Error analyzing {}: {}", file_name, e);
                }
            }
        }
    }

    println!("\n");

    // Summary
    let n_files = file_analyses.len();
    let avg_phrases = if n_files > 0 {
        total_phrases as f64 / n_files as f64
    } else {
        0.0
    };
    let avg_types = if n_files > 0 {
        file_analyses.iter().map(|f| f.n_phrase_types as f64).sum::<f64>() / n_files as f64
    } else {
        0.0
    };
    let avg_entropy = if n_files > 0 {
        file_analyses.iter().map(|f| f.stats.type_entropy).sum::<f64>() / n_files as f64
    } else {
        0.0
    };

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         ANALYSIS SUMMARY                                  ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Dataset: Whistle Signals (192kHz Dolphin/Whale Whistles)                ║");
    println!("║                                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Statistics:                                                              ║");
    println!("║    • Files analyzed:       {:>10}", format!("{}", n_files));
    println!("║    • Total phrases:        {:>10}", format!("{}", total_phrases));
    println!("║    • Unique phrase types:  {:>10}", format!("{}", all_types.len()));
    println!("║    • Avg phrases/file:     {:>10.2}", avg_phrases);
    println!("║    • Avg types/file:       {:>10.2}", avg_types);
    println!("║    • Avg type entropy:     {:>10.3} bits", avg_entropy);
    if errors > 0 {
        println!("║    • Errors:               {:>10}", format!("{}", errors));
    }
    println!("║                                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Analysis Parameters:                                                    ║");
    println!("║    • Min phrase duration:  {:>10.0} ms", config.min_phrase_ms);
    println!("║    • Max phrase duration:  {:>10.0} ms", config.max_phrase_ms);
    println!("║    • Min gap:              {:>10.0} ms", config.min_gap_ms);
    println!("║    • Similarity threshold: {:>10.2}", similarity_threshold);
    println!("║                                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Save results
    let results = DatasetResults {
        dataset: "whistle_signals".to_string(),
        total_files: wav_files.len(),
        files_analyzed: n_files,
        total_phrases,
        unique_phrase_types: all_types.len(),
        file_analyses,
        summary: format!(
            "Analyzed {} whistle recordings, found {} phrases across {} types",
            n_files,
            total_phrases,
            all_types.len()
        ),
    };

    let output_path = output_dir.join("whistle_signals_within_call.json");
    let output_file = File::create(&output_path)?;
    serde_json::to_writer_pretty(BufWriter::new(output_file), &results)?;

    println!("Results saved to: {}", output_path.display());

    Ok(())
}
