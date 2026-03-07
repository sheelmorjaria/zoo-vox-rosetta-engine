// Phase 0: Two-Level HDBSCAN for Egyptian Fruit Bat Phrase Discovery
// OPTIMIZED VERSION: Parallel Processing + Checkpointing + Timestamps
//
// CORRECT ARCHITECTURE:
//   Level 1: Within-Vocalization Segmentation (Energy-based + Change-point detection)
//   Level 2: Cross-Vocalization Vocabulary Building (HDBSCAN on phrase segments)
//
// OPTIMIZATIONS:
//   - Parallel batch processing using rayon
//   - Incremental checkpointing for resume capability
//   - Full timestamp tracking for audio extraction

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use std::time::Instant;
use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use technical_architecture::hdbscan::{DistanceMetric, HdbscanClustering};
use technical_architecture::micro_dynamics_extractor::MicroDynamicsExtractor;
use technical_architecture::pitch::YinEstimator;

// =============================================================================
// Configuration
// =============================================================================

const FRAME_SIZE_MS: usize = 25; // 25ms frames (typical for speech processing)
const FRAME_SHIFT_MS: usize = 10; // 10ms shift (75% overlap)
const MIN_PHRASE_DURATION_MS: usize = 30; // Minimum phrase duration (lowered for more discovery)
const BATCH_SIZE: usize = 100; // Files per batch

// Level 2 HDBSCAN: Cross-vocalization vocabulary
const LEVEL2_MIN_CLUSTER_SIZE: usize = 5; // Minimum phrases per vocabulary item
const LEVEL2_MIN_SAMPLES: usize = 3; // Density threshold for vocabulary

// =============================================================================
// Data Structures with Full Timestamp Info
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrameFeatures {
    frame_index: usize,
    start_time_ms: f64,
    duration_ms: f64,
    features: Vec<f64>, // Multi-dimensional acoustic features
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseSegment {
    segment_id: usize,
    file_index: usize,
    file_name: String,
    start_time_ms: f64,
    end_time_ms: f64,
    duration_ms: f64,
    start_sample: usize, // For audio extraction
    end_sample: usize,   // For audio extraction
    sample_rate: u32,    // For audio extraction
    frame_indices: Vec<usize>,
    level1_cluster_id: i32,
    representative_features: Vec<f64>, // Aggregated features for this phrase
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

// Checkpoint structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessingCheckpoint {
    completed_files: usize,
    total_files: usize,
    all_segments: Vec<PhraseSegment>,
    timestamp: String,
}

// =============================================================================
// Audio Processing
// =============================================================================

fn load_audio_file(file_path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let src = MediaSourceStream::new(Box::new(fs::File::open(file_path)?), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, src, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let sample_rate = track.codec_params.sample_rate.ok_or("Missing sample rate")?;

    let mut audio_samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::ResetRequired) => continue,
            Err(symphonia::core::errors::Error::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(e) => return Err(format!("Failed to read packet: {}", e).into()),
        };

        // Skip packets from other tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                match audio_buf {
                    AudioBufferRef::F32(buf) => {
                        let audio_buffer = buf.as_ref();
                        for plane in audio_buffer.planes().planes() {
                            audio_samples.extend_from_slice(plane);
                            break; // Only first channel
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        let audio_buffer = buf.as_ref();
                        for plane in audio_buffer.planes().planes() {
                            let samples: Vec<f32> = plane.iter().map(|&s| s as f32 / 32768.0).collect();
                            audio_samples.extend(samples);
                            break;
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        let audio_buffer = buf.as_ref();
                        for plane in audio_buffer.planes().planes() {
                            let samples: Vec<f32> = plane.iter().map(|&s| (s as f32 - 128.0) / 128.0).collect();
                            audio_samples.extend(samples);
                            break;
                        }
                    }
                    AudioBufferRef::U16(buf) => {
                        let audio_buffer = buf.as_ref();
                        for plane in audio_buffer.planes().planes() {
                            let samples: Vec<f32> = plane.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).collect();
                            audio_samples.extend(samples);
                            break;
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        let audio_buffer = buf.as_ref();
                        for plane in audio_buffer.planes().planes() {
                            let samples: Vec<f32> = plane.iter().map(|&s| s as f32 / i32::MAX as f32).collect();
                            audio_samples.extend(samples);
                            break;
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        let audio_buffer = buf.as_ref();
                        for plane in audio_buffer.planes().planes() {
                            let samples: Vec<f32> = plane.iter().map(|&s| s as f32).collect();
                            audio_samples.extend(samples);
                            break;
                        }
                    }
                    _ => return Err("Unsupported audio format".into()),
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("Failed to decode packet: {}", e).into()),
        }
    }

    if audio_samples.is_empty() {
        return Err("No audio samples decoded".into());
    }

    Ok((audio_samples, sample_rate))
}

// =============================================================================
// Frame-Level Feature Extraction
// =============================================================================

fn extract_frame_features(audio: &[f32], sample_rate: u32) -> Vec<FrameFeatures> {
    let frame_size_samples = (sample_rate as usize * FRAME_SIZE_MS) / 1000;
    let frame_shift_samples = (sample_rate as usize * FRAME_SHIFT_MS) / 1000;

    let mut frames = Vec::new();
    let mut frame_idx = 0;

    let mut start = 0;
    while start + frame_size_samples <= audio.len() {
        let end = start + frame_size_samples;
        let frame_audio = &audio[start..end];

        let start_time_ms = (start as f64 / sample_rate as f64) * 1000.0;
        let duration_ms = (frame_size_samples as f64 / sample_rate as f64) * 1000.0;

        let features = compute_frame_features(frame_audio, sample_rate);

        frames.push(FrameFeatures {
            frame_index: frame_idx,
            start_time_ms,
            duration_ms,
            features,
        });

        start += frame_shift_samples;
        frame_idx += 1;
    }

    frames
}

fn compute_frame_features(audio: &[f32], sample_rate: u32) -> Vec<f64> {
    let mut features = Vec::new();

    // 1. Energy (RMS)
    let rms = (audio.iter().map(|&x| (x * x) as f64).sum::<f64>() / audio.len() as f64).sqrt();
    features.push(rms.ln_1p()); // Log-scale

    // 2. Zero Crossing Rate
    let zcr = audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64 / audio.len() as f64;
    features.push(zcr);

    // 3. Spectral features
    if let Ok(spectrum) = compute_spectrum(audio) {
        let centroid = compute_spectral_centroid(&spectrum, sample_rate);
        features.push(centroid);

        let rolloff = compute_spectral_rolloff(&spectrum, sample_rate, 0.85);
        features.push(rolloff);

        let bandwidth = compute_spectral_bandwidth(&spectrum, sample_rate, centroid);
        features.push(bandwidth);

        let mel_bands = compute_mel_bands(&spectrum, sample_rate);
        features.extend_from_slice(&mel_bands);
    } else {
        features.extend_from_slice(&[0.0; 13]);
    }

    // 4. Pitch-related features
    if let Some(pitch) = estimate_pitch(audio, sample_rate) {
        features.push(pitch);
        features.push(1.0); // Pitch confidence
    } else {
        features.push(0.0); // No pitch detected
        features.push(0.0); // Zero confidence
    }

    // Pad to 20D (frame-level features for segmentation only)
    while features.len() < 20 {
        features.push(0.0);
    }

    features.truncate(20);
    features
}

fn compute_spectrum(audio: &[f32]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = audio.len();
    let mut spectrum = vec![0.0f64; n / 2];

    let mut windowed = vec![0.0f64; n];
    for (i, &sample) in audio.iter().enumerate() {
        let hann = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
        windowed[i] = sample as f64 * hann;
    }

    for k in 0..n / 2 {
        let mut real = 0.0;
        let mut imag = 0.0;
        for (i, &sample) in windowed.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        spectrum[k] = (real * real + imag * imag).sqrt();
    }

    Ok(spectrum)
}

fn compute_spectral_centroid(spectrum: &[f64], sample_rate: u32) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_mag = 0.0;

    for (i, &mag) in spectrum.iter().enumerate() {
        let freq = i as f64 * sample_rate as f64 / (2.0 * spectrum.len() as f64);
        weighted_sum += freq * mag;
        total_mag += mag;
    }

    if total_mag > 0.0 {
        weighted_sum / total_mag
    } else {
        0.0
    }
}

fn compute_spectral_rolloff(spectrum: &[f64], sample_rate: u32, percentile: f64) -> f64 {
    let total_mag: f64 = spectrum.iter().sum();
    let threshold = total_mag * percentile;

    let mut cumulative = 0.0;
    for (i, &mag) in spectrum.iter().enumerate() {
        cumulative += mag;
        if cumulative >= threshold {
            let freq = i as f64 * sample_rate as f64 / (2.0 * spectrum.len() as f64);
            return freq;
        }
    }

    sample_rate as f64 / 2.0
}

fn compute_spectral_bandwidth(spectrum: &[f64], sample_rate: u32, centroid: f64) -> f64 {
    let mut weighted_dev = 0.0;
    let mut total_mag = 0.0;

    for (i, &mag) in spectrum.iter().enumerate() {
        let freq = i as f64 * sample_rate as f64 / (2.0 * spectrum.len() as f64);
        let dev = (freq - centroid).abs();
        weighted_dev += dev * mag;
        total_mag += mag;
    }

    if total_mag > 0.0 {
        weighted_dev / total_mag
    } else {
        0.0
    }
}

fn compute_mel_bands(spectrum: &[f64], sample_rate: u32) -> Vec<f64> {
    let num_bands = 8;
    let mut bands = vec![0.0; num_bands];

    let nyquist = sample_rate as f64 / 2.0;
    let mel_nyquist = hz_to_mel(nyquist);

    for (i, &mag) in spectrum.iter().enumerate() {
        let freq = (i as f64 / spectrum.len() as f64) * nyquist;
        let mel = hz_to_mel(freq);

        let band_idx = ((mel / mel_nyquist) * num_bands as f64) as usize;
        if band_idx < num_bands {
            bands[band_idx] += mag;
        }
    }

    for band in bands.iter_mut() {
        *band = band.ln_1p();
    }

    bands
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn estimate_pitch(audio: &[f32], sample_rate: u32) -> Option<f64> {
    let min_period = (sample_rate as usize / 500).max(1);
    let max_period = (sample_rate as usize / 50).min(audio.len() / 2);

    let mut best_period = min_period;
    let mut best_corr = 0.0;

    for period in min_period..=max_period {
        let mut correlation = 0.0;
        for i in 0..(audio.len() - period) {
            correlation += audio[i] as f64 * audio[i + period] as f64;
        }
        correlation /= (audio.len() - period) as f64;

        if correlation > best_corr {
            best_corr = correlation;
            best_period = period;
        }
    }

    if best_corr > 0.3 {
        Some(sample_rate as f64 / best_period as f64)
    } else {
        None
    }
}

// =============================================================================
// Level 1: Aggressive Segmentation (Energy + Change-Point)
// =============================================================================

fn segment_by_energy(frames: &[FrameFeatures], sample_rate: u32) -> Vec<usize> {
    if frames.len() < 5 {
        return vec![];
    }

    let mut boundaries = Vec::new();
    let window_size = 5;

    // Compute energy profile
    let mut energy_profile: Vec<f64> = Vec::new();
    for frame in frames {
        let rms = frame.features[0];
        energy_profile.push(rms.exp());
    }

    // Smooth energy profile
    let mut smoothed = Vec::new();
    for i in 0..energy_profile.len() {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(energy_profile.len());
        let avg: f64 = energy_profile[start..end].iter().sum::<f64>() / (end - start) as f64;
        smoothed.push(avg);
    }

    // Find local minima as potential boundaries
    for i in 2..smoothed.len().saturating_sub(2) {
        let prev_avg = smoothed[i - 2..i].iter().sum::<f64>() / 3.0;
        let next_avg = smoothed[i + 1..=(i + 3).min(smoothed.len() - 1)].iter().sum::<f64>() / 3.0;
        let current = smoothed[i];

        if current < prev_avg * 0.5 && current < next_avg * 0.5 {
            if boundaries.last().map_or(true, |&last| i - last > 10) {
                boundaries.push(i);
            }
        }
    }

    boundaries
}

fn segment_by_change_point(frames: &[FrameFeatures]) -> Vec<usize> {
    if frames.len() < 10 {
        return vec![];
    }

    let mut boundaries = Vec::new();
    let window_size = 10;

    for i in window_size..frames.len().saturating_sub(window_size) {
        let mut significant_change = false;

        for feat_idx in 0..frames[0].features.len().min(10) {
            let before: Vec<f64> = frames[i - window_size..i]
                .iter()
                .map(|f| f.features[feat_idx])
                .collect();

            let after: Vec<f64> = frames[i..i + window_size]
                .iter()
                .map(|f| f.features[feat_idx])
                .collect();

            let mean_before = before.iter().sum::<f64>() / before.len() as f64;
            let mean_after = after.iter().sum::<f64>() / after.len() as f64;

            let var_before = before.iter().map(|&x| (x - mean_before).powi(2)).sum::<f64>() / before.len() as f64;
            let var_after = after.iter().map(|&x| (x - mean_after).powi(2)).sum::<f64>() / after.len() as f64;

            let pooled_std = ((var_before + var_after) / 2.0).sqrt();
            if pooled_std > 0.001 {
                let z_score = (mean_after - mean_before).abs() / pooled_std;
                if z_score > 2.0 {
                    // Lowered threshold for more boundaries
                    significant_change = true;
                    break;
                }
            }
        }

        if significant_change {
            if boundaries.last().map_or(true, |&last| i - last > 15) {
                boundaries.push(i);
            }
        }
    }

    boundaries
}

fn combine_segmentation_methods(frames: &[FrameFeatures], sample_rate: u32) -> Vec<(usize, usize)> {
    let energy_bounds = segment_by_energy(frames, sample_rate);
    let change_bounds = segment_by_change_point(frames);

    let mut combined: Vec<usize> = energy_bounds.into_iter().chain(change_bounds.into_iter()).collect();

    combined.sort();
    combined.dedup();

    // Filter for minimum spacing
    let min_spacing = 10;
    let mut filtered = Vec::new();
    for boundary in combined {
        if filtered.last().map_or(true, |&last| boundary - last >= min_spacing) {
            filtered.push(boundary);
        }
    }

    // Convert to segments
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;

    for &boundary in &filtered {
        if boundary - start >= 3 {
            segments.push((start, boundary));
            start = boundary;
        }
    }

    if frames.len() - start >= 3 {
        segments.push((start, frames.len()));
    }

    segments
}

/// Process a single vocalization and extract phrase segments
fn process_single_vocalization(
    file_idx: usize,
    file_path: &Path,
) -> Result<Vec<PhraseSegment>, Box<dyn std::error::Error>> {
    let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown.wav");

    // Load audio
    let (audio, sample_rate) = load_audio_file(file_path)?;

    // Extract frames
    let frames = extract_frame_features(&audio, sample_rate);
    if frames.len() < 3 {
        return Ok(vec![]);
    }

    // Segment using aggressive methods
    let segment_ranges = combine_segmentation_methods(&frames, sample_rate);

    // Initialize 30D microdynamics extractor
    let md_extractor = MicroDynamicsExtractor::new(sample_rate);

    // Convert to PhraseSegments with full timestamp info
    let mut segments = Vec::new();

    for (seg_idx, (start, end)) in segment_ranges.iter().enumerate() {
        let start_frame = &frames[*start];
        let end_frame = &frames[end.saturating_sub(1)];

        // Calculate timestamps and sample positions for audio extraction
        let start_time_ms = start_frame.start_time_ms;
        let end_time_ms = end_frame.start_time_ms + end_frame.duration_ms;
        let duration_ms = end_time_ms - start_time_ms;

        // Duration filter
        if duration_ms < MIN_PHRASE_DURATION_MS as f64 {
            continue;
        }

        // Calculate sample positions for audio extraction
        let start_sample = ((start_time_ms / 1000.0) * sample_rate as f64) as usize;
        let end_sample = ((end_time_ms / 1000.0) * sample_rate as f64) as usize;

        // Extract 30D microdynamics features from the actual phrase audio
        let phrase_audio = &audio[start_sample..end_sample];
        let rep_features = match md_extractor.extract(phrase_audio) {
            Ok(md_features) => {
                // Convert to Vector30D and then to flat Vec<f64> using YIN F0 estimation
                let mean_f0 = estimate_f0_from_audio(phrase_audio, sample_rate);
                let f0_range = estimate_f0_range_from_audio(phrase_audio, sample_rate);
                let vec30d = md_features.to_vector30d(mean_f0, duration_ms as f32, f0_range);
                vector30d_to_vec(vec30d)
            }
            Err(_) => {
                // Fallback to aggregated frame features if microdynamics extraction fails
                let mut fallback = vec![0.0; 30];
                let frame_indices: Vec<usize> = (*start..*end).collect();
                for &idx in &frame_indices {
                    for (f_idx, &feat) in frames[idx].features.iter().enumerate().take(20) {
                        fallback[f_idx] += feat;
                    }
                }
                for feat in fallback.iter_mut().take(20) {
                    *feat /= frame_indices.len().max(1) as f64;
                }
                // Pad remaining dimensions with defaults
                fallback[0] = 7000.0; // mean_f0_hz
                fallback[1] = 400.0; // f0_range_hz
                fallback[2] = duration_ms; // duration_ms
                fallback
            }
        };

        let frame_indices: Vec<usize> = (*start..*end).collect();

        segments.push(PhraseSegment {
            segment_id: seg_idx,
            file_index: file_idx,
            file_name: file_name.to_string(),
            start_time_ms,
            end_time_ms,
            duration_ms,
            start_sample,
            end_sample,
            sample_rate,
            frame_indices,
            level1_cluster_id: seg_idx as i32,
            representative_features: rep_features,
        });
    }

    Ok(segments)
}

// =============================================================================
// 30D Microdynamics Helper Functions
// =============================================================================

/// Convert Vector30D to flat Vec<f64> for HDBSCAN clustering
fn vector30d_to_vec(v: technical_architecture::island_hopping::Vector30D) -> Vec<f64> {
    vec![
        // Fundamental (3)
        v.mean_f0_hz as f64,
        v.f0_range_hz as f64,
        v.duration_ms as f64,
        // Grit Factors (3)
        v.harmonic_to_noise_ratio as f64,
        v.spectral_flatness as f64,
        v.harmonicity as f64,
        // Motion Factors (7)
        v.attack_time_ms as f64,
        v.decay_time_ms as f64,
        v.sustain_level as f64,
        v.vibrato_rate_hz as f64,
        v.vibrato_depth as f64,
        v.jitter as f64,
        v.shimmer as f64,
        // Fingerprint Factors (13 MFCCs)
        v.mfcc_1 as f64,
        v.mfcc_2 as f64,
        v.mfcc_3 as f64,
        v.mfcc_4 as f64,
        v.mfcc_5 as f64,
        v.mfcc_6 as f64,
        v.mfcc_7 as f64,
        v.mfcc_8 as f64,
        v.mfcc_9 as f64,
        v.mfcc_10 as f64,
        v.mfcc_11 as f64,
        v.mfcc_12 as f64,
        v.mfcc_13 as f64,
        // Spectral Dynamics (1)
        v.spectral_flux as f64,
        // Rhythm Factors (3)
        v.median_ici_ms as f64,
        v.onset_rate_hz as f64,
        v.ici_coefficient_of_variation as f64,
    ]
}

/// Estimate mean F0 from audio using YIN algorithm
fn estimate_f0_from_audio(audio: &[f32], sample_rate: u32) -> f32 {
    // Configure YIN for bat vocalizations (typically 5-15 kHz range for Egyptian fruit bats)
    let yin = YinEstimator::with_range(sample_rate, 5000.0, 15000.0);
    let (f0, confidence) = yin.estimate(audio);

    // Use default bat F0 if confidence is too low
    if confidence > 0.3 && f0 > 0.0 {
        f0
    } else {
        7000.0 // Default bat F0 if no clear pitch detected
    }
}

/// Estimate F0 range from audio (min to max) using YIN
fn estimate_f0_range_from_audio(audio: &[f32], sample_rate: u32) -> f32 {
    // Split into windows and estimate F0 for each using YIN
    let window_size = (sample_rate as usize * 10) / 1000; // 10ms windows
    let mut f0_values = Vec::new();

    let yin = YinEstimator::with_range(sample_rate, 5000.0, 15000.0);

    for i in (0..audio.len().saturating_sub(window_size)).step_by(window_size) {
        let window = &audio[i..(i + window_size).min(audio.len())];
        let (f0, confidence) = yin.estimate(window);
        if confidence > 0.3 && f0 > 1000.0 && f0 < 20000.0 {
            f0_values.push(f0);
        }
    }

    if f0_values.is_empty() {
        return 400.0; // Default F0 range
    }

    let min_f0 = f0_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_f0 = f0_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    max_f0 - min_f0
}

// =============================================================================
// Checkpointing Functions
// =============================================================================

fn save_checkpoint(results_dir: &Path, checkpoint: &ProcessingCheckpoint) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_path = results_dir.join("checkpoint.json");
    let json = serde_json::to_string_pretty(checkpoint)?;
    fs::write(&checkpoint_path, json)?;
    Ok(())
}

fn load_checkpoint(results_dir: &Path) -> Result<Option<ProcessingCheckpoint>, Box<dyn std::error::Error>> {
    let checkpoint_path = results_dir.join("checkpoint.json");
    if !checkpoint_path.exists() {
        return Ok(None);
    }

    let json = fs::read_to_string(&checkpoint_path)?;
    let checkpoint: ProcessingCheckpoint = serde_json::from_str(&json)?;
    Ok(Some(checkpoint))
}

fn load_existing_segments(results_dir: &Path) -> Vec<PhraseSegment> {
    match load_checkpoint(results_dir) {
        Ok(Some(checkpoint)) => {
            println!("   📂 Found checkpoint: {} files completed", checkpoint.completed_files);
            checkpoint.all_segments
        }
        _ => vec![],
    }
}

// =============================================================================
// Level 2: Cross-Vocalization Vocabulary Building
// =============================================================================

fn build_vocabulary(
    all_segments: &[PhraseSegment],
    hdbscan: &HdbscanClustering,
) -> Result<Vec<VocabularyItem>, Box<dyn std::error::Error>> {
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Level 2: Cross-Vocalization Vocabulary Building                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!(
        "   📊 Total phrase segments from all vocalizations: {}",
        all_segments.len()
    );

    if all_segments.is_empty() {
        println!("   ⚠ No phrase segments found!");
        return Ok(vec![]);
    }

    let n_segments = all_segments.len();
    let n_features = all_segments[0].representative_features.len();

    println!(
        "   📊 Feature dimensions: {} segments × {}D features",
        n_segments, n_features
    );

    let mut flat_features = Vec::with_capacity(n_segments * n_features);
    for segment in all_segments {
        flat_features.extend_from_slice(&segment.representative_features);
    }

    let features_array = Array2::from_shape_vec((n_segments, n_features), flat_features)?;

    println!("   🔍 Running Level 2 HDBSCAN (cross-vocalization)...");
    let vocab_start = Instant::now();

    let labels = hdbscan.fit_predict_hnsw(&features_array)?;

    let vocab_time = vocab_start.elapsed();
    println!("   ✅ Vocabulary built in {:.2}s", vocab_time.as_secs_f64());

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

    println!("   📚 Discovered {} vocabulary items", vocabulary.len());

    Ok(vocabulary)
}

// =============================================================================
// Main with Parallel Processing and Checkpointing
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Phase 0: Two-Level HDBSCAN - OPTIMIZED (Parallel + Checkpointing)        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  • Parallel batch processing with rayon                                    ║");
    println!("║  • Incremental checkpointing for resume capability                           ║");
    println!("║  • Full timestamps for audio extraction                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    // Use test directory for initial testing - comment out to use full dataset
    let audio_dir = if Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio_test").exists() {
        data_dir.join("audio_test")
    } else {
        data_dir.join("audio")
    };
    let results_dir = data_dir.join("phase0_twolevel_hdbscan_results");

    fs::create_dir_all(&results_dir)?;

    // Discover WAV files
    let mut wav_files: Vec<_> = fs::read_dir(&audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map(|e| e == "wav").unwrap_or(false))
        .map(|entry| entry.path())
        .collect();

    // Sort by file name for reproducibility
    wav_files.sort_by_key(|a| a.to_string_lossy().to_string());

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Dataset Overview                                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("   📂 Audio Directory: {}", audio_dir.display());
    println!("   🔢 Total WAV files: {}", wav_files.len());
    println!("   💾 Results Directory: {}", results_dir.display());
    println!();

    if wav_files.is_empty() {
        println!("   ❌ No WAV files found!");
        return Ok(());
    }

    // Check for existing checkpoint
    let existing_segments = load_existing_segments(&results_dir);
    let start_idx = existing_segments.len() / BATCH_SIZE;

    if start_idx > 0 {
        println!("   📂 Resuming from batch {}", start_idx);
        wav_files = wav_files.into_iter().skip(start_idx * BATCH_SIZE).collect();
    }

    // Initialize Level 2 HDBSCAN with Cosine distance metric
    // Cosine focuses on spectral pattern/shape rather than absolute F0 values
    let level2_hdbscan =
        HdbscanClustering::with_metric(LEVEL2_MIN_CLUSTER_SIZE, LEVEL2_MIN_SAMPLES, DistanceMetric::Cosine)?;

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Level 1: Parallel Within-Vocalization Segmentation                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("   🏗️  Aggressive Segmentation Configuration:");
    println!("      ├─ Methods: Energy-based + Change-point detection");
    println!(
        "      ├─ Frame size: {}ms, Frame shift: {}ms",
        FRAME_SIZE_MS, FRAME_SHIFT_MS
    );
    println!("      ├─ Min phrase duration: {}ms", MIN_PHRASE_DURATION_MS);
    println!("      ├─ Batch size: {} files", BATCH_SIZE);
    println!("      └─ Energy threshold: 50% drop, Z-score > 2.0");
    println!();

    let level1_start = Instant::now();

    // Process in batches
    let mut all_segments = existing_segments;
    let total_batches = (wav_files.len() + BATCH_SIZE - 1) / BATCH_SIZE;

    for batch_idx in 0..total_batches {
        let batch_start = batch_idx * BATCH_SIZE;
        let batch_end = (batch_start + BATCH_SIZE).min(wav_files.len());
        let batch_files: Vec<_> = wav_files[batch_start..batch_end].to_vec();

        println!(
            "   🔄 Batch {}/{} (files {}-{})...",
            batch_idx + 1,
            total_batches,
            batch_start,
            batch_end - 1
        );

        // Process batch in parallel
        let batch_segments: Vec<Vec<PhraseSegment>> = batch_files
            .par_iter()
            .enumerate()
            .map(|(local_idx, file_path)| {
                let file_idx = batch_start + local_idx;
                process_single_vocalization(file_idx, file_path).unwrap_or_else(|e| {
                    eprintln!("        ⚠ Failed to process {:?}: {}", file_path, e);
                    vec![]
                })
            })
            .collect();

        // Flatten and collect results
        let mut batch_segment_count = 0;
        for mut segments in batch_segments {
            batch_segment_count += segments.len();
            all_segments.append(&mut segments);
        }

        println!(
            "      📝 Discovered {} phrase segments (total: {})",
            batch_segment_count,
            all_segments.len()
        );

        // Save checkpoint after each batch
        let checkpoint = ProcessingCheckpoint {
            completed_files: batch_end,
            total_files: wav_files.len() + start_idx * BATCH_SIZE,
            all_segments: all_segments.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        save_checkpoint(&results_dir, &checkpoint)?;
        println!("      💾 Checkpoint saved");
        println!();
    }

    let level1_time = level1_start.elapsed();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Level 1 Complete                                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("   ✅ Processed {} vocalizations", all_segments.len());
    println!("   📝 Total phrase segments discovered: {}", all_segments.len());
    println!("   ⏱️  Level 1 time: {:.2}s", level1_time.as_secs_f64());
    println!();

    // Build vocabulary
    let vocabulary = build_vocabulary(&all_segments, &level2_hdbscan)?;

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Vocabulary Statistics                                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    if vocabulary.is_empty() {
        println!("   ⚠ No vocabulary items discovered (all phrases classified as noise)");
    } else {
        println!("   📚 Vocabulary Size: {} items", vocabulary.len());
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
    }

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Saving Results                                                           │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save vocabulary
    let vocab_path = results_dir.join("vocabulary.json");
    let vocab_json = serde_json::to_string_pretty(&vocabulary)?;
    fs::write(&vocab_path, vocab_json)?;
    println!("   💾 Vocabulary: {}", vocab_path.display());

    // Save all segments with timestamps
    let segments_path = results_dir.join("all_segments.json");
    let segments_json = serde_json::to_string_pretty(&all_segments)?;
    fs::write(&segments_path, segments_json)?;
    println!("   💾 All segments: {}", segments_path.display());

    // Save timestamp map for audio extraction
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
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Symbolic Stream Generation                                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Generate symbolic stream with cluster IDs
    let mut symbolic_stream: Vec<i32> = Vec::new();
    let mut segment_to_vocab: std::collections::HashMap<usize, i32> = std::collections::HashMap::new();

    // Build mapping from segment to vocabulary ID
    for vocab in &vocabulary {
        for phrase in &vocab.example_phrases {
            segment_to_vocab.insert(phrase.segment_id, vocab.vocabulary_id as i32);
        }
    }

    // Generate stream by sorting segments by file_index and segment_id
    let mut sorted_segments: Vec<_> = all_segments.iter().collect();
    sorted_segments.sort_by(|a, b| {
        a.file_index
            .cmp(&b.file_index)
            .then_with(|| a.segment_id.cmp(&b.segment_id))
    });

    for segment in sorted_segments {
        let vocab_id = segment_to_vocab.get(&segment.segment_id).copied().unwrap_or(-1); // -1 for noise/unclassified
        symbolic_stream.push(vocab_id);
    }

    println!("   📝 Symbolic Stream Statistics:");
    println!("      ├─ Total symbols: {}", symbolic_stream.len());
    println!("      ├─ Unique vocabulary items: {}", vocabulary.len());
    println!(
        "      └─ Noise symbols: {}",
        symbolic_stream.iter().filter(|&&x| x == -1).count()
    );

    // Save symbolic stream
    let stream_path = results_dir.join("symbolic_stream.txt");
    let stream_text: String = symbolic_stream
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&stream_path, stream_text)?;
    println!("   💾 Symbolic stream: {}", stream_path.display());

    // Show preview
    println!();
    println!("   🔤 Symbolic Stream Preview (first 50 symbols):");
    println!(
        "      └─ {}",
        symbolic_stream
            .iter()
            .take(50)
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE 0 COMPLETE - OPTIMIZED VERSION                       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Parallel + Checkpointed Two-Level HDBSCAN                              ║");
    println!("║                                                                           ║");
    println!("║  SUMMARY:                                                                 ║");
    println!(
        "║     • Vocalizations processed: {}                                         ║",
        wav_files.len()
    );
    println!(
        "║     • Phrase segments discovered: {}                                     ║",
        all_segments.len()
    );
    println!(
        "║     • Vocabulary items discovered: {}                                     ║",
        vocabulary.len()
    );
    println!(
        "║     • Symbolic stream length: {} symbols                                 ║",
        symbolic_stream.len()
    );
    println!("║                                                                           ║");
    println!("║  OUTPUT FILES:                                                           ║");
    println!("║     • vocabulary.json - Vocabulary items with metadata                    ║");
    println!("║     • all_segments.json - All phrase segments with timestamps            ║");
    println!("║     • timestamp_map.json - Audio extraction info                          ║");
    println!("║     • symbolic_stream.txt - Symbol sequence (cluster IDs)                 ║");
    println!("║     • checkpoint.json - Resume checkpoint                                 ║");
    println!("║                                                                           ║");
    println!("║  AUDIO EXTRACTION:                                                       ║");
    println!("║     1. Load timestamp_map.json                                            ║");
    println!("║     2. Select vocabulary_id to extract                                    ║");
    println!("║     3. For each segment:                                                 ║");
    println!("║        - Load WAV file                                                   ║");
    println!("║        - Seek to start_sample                                             ║");
    println!("║        - Extract (end_sample - start_sample) samples                      ║");
    println!("║                                                                           ║");
    println!("║  NEXT STEPS:                                                             ║");
    println!("║     • Analyze vocabulary item distributions                              ║");
    println!("║     • Extract audio for each vocabulary item                              ║");
    println!("║     • Build phrase transition statistics                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
