// Phase 0: Two-Level HDBSCAN for Egyptian Fruit Bat Phrase Discovery
//
// CORRECT ARCHITECTURE:
//   Level 1: Within-Vocalization Segmentation (HDBSCAN on frame-level features)
//   Level 2: Cross-Vocalization Vocabulary Building (HDBSCAN on phrase segments)
//
// Each WAV file is a "sentence" containing multiple "phrases" (words).
// We discover phrases WITHIN each vocalization, then build a vocabulary ACROSS vocalizations.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
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
use technical_architecture::hdbscan::HdbscanClustering;

// =============================================================================
// Configuration
// =============================================================================

const FRAME_SIZE_MS: usize = 25; // 25ms frames (typical for speech processing)
const FRAME_SHIFT_MS: usize = 10; // 10ms shift (75% overlap)
const MIN_PHRASE_DURATION_MS: usize = 50; // Minimum phrase duration
const MAX_PHRASE_DURATION_MS: usize = 500; // Maximum phrase duration

// Level 1 HDBSCAN: Within-vocalization segmentation
const LEVEL1_MIN_CLUSTER_SIZE: usize = 30; // Minimum frames per phrase
const LEVEL1_MIN_SAMPLES: usize = 10; // Density threshold for phrase detection

// Level 2 HDBSCAN: Cross-vocalization vocabulary
const LEVEL2_MIN_CLUSTER_SIZE: usize = 5; // Minimum phrases per vocabulary item (lowered for small datasets)
const LEVEL2_MIN_SAMPLES: usize = 3; // Density threshold for vocabulary (lowered for small datasets)

// =============================================================================
// Data Structures
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

    // 3. Spectral features (simplified FFT-based)
    if let Ok(spectrum) = compute_spectrum(audio) {
        // Spectral centroid
        let centroid = compute_spectral_centroid(&spectrum, sample_rate);
        features.push(centroid);

        // Spectral rolloff (85%)
        let rolloff = compute_spectral_rolloff(&spectrum, sample_rate, 0.85);
        features.push(rolloff);

        // Spectral bandwidth
        let bandwidth = compute_spectral_bandwidth(&spectrum, sample_rate, centroid);
        features.push(bandwidth);

        // MFCC-like features (using log mel bands)
        let mel_bands = compute_mel_bands(&spectrum, sample_rate);
        features.extend_from_slice(&mel_bands);
    } else {
        // Add zeros if FFT fails
        features.extend_from_slice(&[0.0; 13]);
    }

    // 4. Temporal features
    // Energy delta (difference from previous frame would be computed externally)
    // For now, we'll add some basic temporal statistics

    // 5. Pitch-related features (simplified)
    if let Some(pitch) = estimate_pitch(audio, sample_rate) {
        features.push(pitch);
        features.push(1.0); // Pitch confidence
    } else {
        features.push(0.0); // No pitch detected
        features.push(0.0); // Zero confidence
    }

    // Total: 1 (RMS) + 1 (ZCR) + 3 (spectral) + 8 (mel bands) + 2 (pitch) = 15 features
    // Let's pad to 20D for consistency with 30D features
    while features.len() < 20 {
        features.push(0.0);
    }

    features.truncate(20);
    features
}

fn compute_spectrum(audio: &[f32]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Simple power spectrum using windowed FFT
    // For now, use a simplified approach
    let n = audio.len();
    let mut spectrum = vec![0.0f64; n / 2];

    // Apply Hann window
    let mut windowed = vec![0.0f64; n];
    for (i, &sample) in audio.iter().enumerate() {
        let hann = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
        windowed[i] = sample as f64 * hann;
    }

    // Naive DFT (for simplicity - in production use FFT)
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
    // Compute 8 mel-frequency band energies
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

    // Log-scale and normalize
    for band in bands.iter_mut() {
        *band = band.ln_1p();
    }

    bands
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn estimate_pitch(audio: &[f32], sample_rate: u32) -> Option<f64> {
    // Simplified autocorrelation-based pitch estimation
    let min_period = (sample_rate as usize / 500).max(1); // Max 500 Hz
    let max_period = (sample_rate as usize / 50).min(audio.len() / 2); // Min 50 Hz

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

    // Only return pitch if correlation is strong enough
    if best_corr > 0.3 {
        Some(sample_rate as f64 / best_period as f64)
    } else {
        None
    }
}

// =============================================================================
// Level 1: Within-Vocalization Segmentation
// =============================================================================

/// Energy-based segmentation - detects phrase boundaries based on energy changes
fn segment_by_energy(frames: &[FrameFeatures], sample_rate: u32) -> Vec<usize> {
    if frames.len() < 5 {
        return vec![];
    }

    let mut boundaries = Vec::new();
    let window_size = 5; // frames to average over

    // Compute energy profile
    let mut energy_profile: Vec<f64> = Vec::new();
    for frame in frames {
        // RMS energy is the first feature
        let rms = frame.features[0];
        energy_profile.push(rms.exp()); // Convert from log scale back to linear
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

        // Local minimum with significant drop
        if current < prev_avg * 0.5 && current < next_avg * 0.5 {
            // Check minimum spacing between boundaries
            if boundaries.last().map_or(true, |&last| i - last > 10) {
                boundaries.push(i);
            }
        }
    }

    boundaries
}

/// Statistical change-point detection using sliding window
fn segment_by_change_point(frames: &[FrameFeatures]) -> Vec<usize> {
    if frames.len() < 10 {
        return vec![];
    }

    let mut boundaries = Vec::new();
    let window_size = 10;

    // For each feature, compute statistics before and after each point
    for i in window_size..frames.len().saturating_sub(window_size) {
        let mut significant_change = false;

        for feat_idx in 0..frames[0].features.len().min(10) {
            // Use first 10 features
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

            let std_before = var_before.sqrt();
            let std_after = var_after.sqrt();

            // Check if means differ significantly (using pooled std)
            let pooled_std = ((var_before + var_after) / 2.0).sqrt();
            if pooled_std > 0.001 {
                let z_score = (mean_after - mean_before).abs() / pooled_std;
                if z_score > 2.5 {
                    // Significant change threshold
                    significant_change = true;
                    break;
                }
            }
        }

        if significant_change {
            // Check minimum spacing between boundaries
            if boundaries.last().map_or(true, |&last| i - last > 15) {
                boundaries.push(i);
            }
        }
    }

    boundaries
}

/// Combine multiple segmentation methods using voting
fn combine_segmentation_methods(frames: &[FrameFeatures], sample_rate: u32) -> Vec<(usize, usize)> {
    let all_boundaries: Vec<usize> = Vec::new();

    // Get boundaries from different methods
    let energy_bounds = segment_by_energy(frames, sample_rate);
    let change_bounds = segment_by_change_point(frames);

    println!("        📊 Energy-based boundaries: {}", energy_bounds.len());
    println!("        📊 Change-point boundaries: {}", change_bounds.len());

    // Combine with minimum spacing
    let mut combined: Vec<usize> = energy_bounds.into_iter().chain(change_bounds.into_iter()).collect();

    combined.sort();
    combined.dedup();

    // Filter for minimum spacing
    let min_spacing = 10; // minimum frames between boundaries
    let mut filtered = Vec::new();
    for boundary in combined {
        if filtered.last().map_or(true, |&last| boundary - last >= min_spacing) {
            filtered.push(boundary);
        }
    }

    // Convert to segments (start, end) pairs
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;

    for &boundary in &filtered {
        if boundary - start >= 3 {
            // Minimum 3 frames per segment
            segments.push((start, boundary));
            start = boundary;
        }
    }

    // Add final segment
    if frames.len() - start >= 3 {
        segments.push((start, frames.len()));
    }

    println!("        📝 Combined into {} segments", segments.len());

    segments
}

fn segment_vocalization_hdbscan(
    file_name: &str,
    file_index: usize,
    audio: &[f32],
    sample_rate: u32,
    hdbscan: &HdbscanClustering,
) -> Result<Vec<PhraseSegment>, Box<dyn std::error::Error>> {
    println!(
        "      🎵 Segmenting: {} ({} samples, {} Hz)",
        file_name,
        audio.len(),
        sample_rate
    );

    // Extract frame-level features
    let frames = extract_frame_features(audio, sample_rate);
    if frames.len() < LEVEL1_MIN_CLUSTER_SIZE {
        println!("        ⚠ Too few frames ({})", frames.len());
        return Ok(vec![]);
    }

    println!("        📊 Extracted {} frames", frames.len());

    // Build feature matrix for HDBSCAN
    let mut feature_matrix = Vec::with_capacity(frames.len());
    for frame in &frames {
        feature_matrix.push(frame.features.clone());
    }

    let n_frames = feature_matrix.len();
    let n_features = feature_matrix[0].len();

    // Convert to Array2
    let mut flat_features = Vec::with_capacity(n_frames * n_features);
    for frame_features in &feature_matrix {
        flat_features.extend_from_slice(frame_features);
    }

    let features_array = Array2::from_shape_vec((n_frames, n_features), flat_features)
        .map_err(|e| format!("Failed to create feature array: {}", e))?;

    // Run HDBSCAN to discover phrase clusters within this vocalization
    println!("        🔍 Running Level 1 HDBSCAN (within-vocalization)...");
    let cluster_start = Instant::now();

    let labels = hdbscan.fit_predict_hnsw(&features_array)?;

    let cluster_time = cluster_start.elapsed();
    println!("        ✅ Segmentation complete in {:.2}s", cluster_time.as_secs_f64());

    // Convert frame-level clusters to phrase segments
    let mut segments = Vec::new();
    let mut segment_id = 0;

    // Group consecutive frames with the same cluster ID
    let mut current_cluster = labels.first().copied().unwrap_or(-1);
    let mut segment_start = 0;

    for (i, &label) in labels.iter().enumerate() {
        if label != current_cluster || i == labels.len() - 1 {
            // End of current segment
            let end_idx = if label != current_cluster { i } else { i + 1 };

            // Filter out noise segments (-1) and very short/long segments
            if current_cluster >= 0 {
                let start_frame = &frames[segment_start];
                let end_frame = &frames[end_idx.saturating_sub(1)];
                let duration_ms = end_frame.start_time_ms + end_frame.duration_ms - start_frame.start_time_ms;

                if duration_ms >= MIN_PHRASE_DURATION_MS as f64 && duration_ms <= MAX_PHRASE_DURATION_MS as f64 {
                    // Aggregate features for this segment (mean of frame features)
                    let mut rep_features = vec![0.0; n_features];
                    let frame_indices: Vec<usize> = (segment_start..end_idx).collect();

                    for &idx in &frame_indices {
                        for (f_idx, &feat) in frames[idx].features.iter().enumerate() {
                            rep_features[f_idx] += feat;
                        }
                    }

                    for feat in rep_features.iter_mut() {
                        *feat /= frame_indices.len() as f64;
                    }

                    segments.push(PhraseSegment {
                        segment_id,
                        file_index,
                        file_name: file_name.to_string(),
                        start_time_ms: start_frame.start_time_ms,
                        end_time_ms: end_frame.start_time_ms + end_frame.duration_ms,
                        duration_ms,
                        frame_indices,
                        level1_cluster_id: current_cluster,
                        representative_features: rep_features,
                    });

                    segment_id += 1;
                }
            }

            current_cluster = label;
            segment_start = i;
        }
    }

    println!("        📝 Discovered {} phrase segments", segments.len());

    Ok(segments)
}

/// Aggressive segmentation using energy-based and change-point detection
/// This method detects phrase boundaries based on acoustic changes rather than clustering
fn segment_vocalization_aggressive(
    file_name: &str,
    file_index: usize,
    audio: &[f32],
    sample_rate: u32,
) -> Result<Vec<PhraseSegment>, Box<dyn std::error::Error>> {
    println!(
        "      🎵 Segmenting (AGGRESSIVE): {} ({} samples, {} Hz)",
        file_name,
        audio.len(),
        sample_rate
    );

    // Extract frame-level features
    let frames = extract_frame_features(audio, sample_rate);
    if frames.len() < 5 {
        println!("        ⚠ Too few frames ({})", frames.len());
        return Ok(vec![]);
    }

    println!("        📊 Extracted {} frames", frames.len());

    // Use combined segmentation methods
    let segment_ranges = combine_segmentation_methods(&frames, sample_rate);

    // Convert segment ranges to PhraseSegments
    let mut segments = Vec::new();
    let n_features = frames[0].features.len();

    for (seg_idx, (start, end)) in segment_ranges.iter().enumerate() {
        let start_frame = &frames[*start];
        let end_frame = &frames[end.saturating_sub(1)];
        let duration_ms = end_frame.start_time_ms + end_frame.duration_ms - start_frame.start_time_ms;

        // Duration filter
        if duration_ms < MIN_PHRASE_DURATION_MS as f64 {
            continue; // Too short
        }

        // Aggregate features for this segment (mean of frame features)
        let mut rep_features = vec![0.0; n_features];
        let frame_indices: Vec<usize> = (*start..*end).collect();

        for &idx in &frame_indices {
            for (f_idx, &feat) in frames[idx].features.iter().enumerate() {
                rep_features[f_idx] += feat;
            }
        }

        for feat in rep_features.iter_mut() {
            *feat /= frame_indices.len() as f64;
        }

        segments.push(PhraseSegment {
            segment_id: seg_idx,
            file_index,
            file_name: file_name.to_string(),
            start_time_ms: start_frame.start_time_ms,
            end_time_ms: end_frame.start_time_ms + end_frame.duration_ms,
            duration_ms,
            frame_indices,
            level1_cluster_id: seg_idx as i32, // Each segment gets unique ID
            representative_features: rep_features,
        });
    }

    println!("        ✅ Discovered {} phrase segments", segments.len());

    Ok(segments)
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

    // Build feature matrix from all phrase segments
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

    let features_array = Array2::from_shape_vec((n_segments, n_features), flat_features)
        .map_err(|e| format!("Failed to create vocabulary feature array: {}", e))?;

    // Run HDBSCAN to build vocabulary
    println!("   🔍 Running Level 2 HDBSCAN (cross-vocalization)...");
    let vocab_start = Instant::now();

    let labels = hdbscan.fit_predict_hnsw(&features_array)?;

    let vocab_time = vocab_start.elapsed();
    println!("   ✅ Vocabulary built in {:.2}s", vocab_time.as_secs_f64());

    // Group segments by vocabulary item (cluster)
    let mut cluster_map: std::collections::HashMap<i32, Vec<&PhraseSegment>> = std::collections::HashMap::new();

    for (segment_idx, &label) in labels.iter().enumerate() {
        if label >= 0 {
            // Skip noise
            cluster_map
                .entry(label)
                .or_insert_with(Vec::new)
                .push(&all_segments[segment_idx]);
        }
    }

    // Build vocabulary items
    let mut vocabulary = Vec::new();
    let mut vocab_id = 0;

    let mut cluster_ids: Vec<_> = cluster_map.keys().cloned().collect();
    cluster_ids.sort();

    for &cluster_id in &cluster_ids {
        let segments = cluster_map.get(&cluster_id).unwrap();

        // Compute statistics
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
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Phase 0: Two-Level HDBSCAN - Bat Phrase Discovery                   ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Level 1: Within-vocalization phrase segmentation                        ║");
    println!("║  Level 2: Cross-vocalization vocabulary building                         ║");
    println!("║                                                                           ║");
    println!("║  Each WAV file is a 'sentence' containing multiple 'phrases' (words)      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let results_dir = data_dir.join("phase0_twolevel_hdbscan_results");

    fs::create_dir_all(&results_dir)?;

    // Discover WAV files
    let wav_files: Vec<_> = fs::read_dir(&audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map(|e| e == "wav").unwrap_or(false))
        .map(|entry| entry.path())
        .collect();

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

    // Process a subset for testing (first 100 files)
    let test_subset_size = std::env::var("TEST_SUBSET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let wav_files: Vec<_> = wav_files.into_iter().take(test_subset_size).collect();

    println!("   🔬 Processing subset of {} files for testing", wav_files.len());
    println!("      (Set TEST_SUBSET=N to process N files)");
    println!();

    // Initialize HDBSCAN for Level 1 (within-vocalization)
    let level1_hdbscan = HdbscanClustering::new(LEVEL1_MIN_CLUSTER_SIZE, LEVEL1_MIN_SAMPLES)?;

    // Initialize HDBSCAN for Level 2 (cross-vocalization vocabulary)
    let level2_hdbscan = HdbscanClustering::new(LEVEL2_MIN_CLUSTER_SIZE, LEVEL2_MIN_SAMPLES)?;

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Level 1: Within-Vocalization Segmentation                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("   🏗️  Level 1: Aggressive Segmentation Configuration:");
    println!("      ├─ Methods: Energy-based + Change-point detection");
    println!("      ├─ Frame size: {}ms (25ms typical for speech)", FRAME_SIZE_MS);
    println!("      ├─ Frame shift: {}ms (75% overlap)", FRAME_SHIFT_MS);
    println!("      ├─ Min phrase duration: {}ms", MIN_PHRASE_DURATION_MS);
    println!("      ├─ Energy threshold: 50% drop = boundary");
    println!("      └─ Change-point threshold: Z-score > 2.5");
    println!();

    let level1_start = Instant::now();

    // Process each vocalization and extract phrase segments
    let mut all_segments: Vec<PhraseSegment> = Vec::new();
    let mut total_frames_processed = 0;

    for (file_idx, wav_file) in wav_files.iter().enumerate() {
        let file_name = wav_file.file_name().and_then(|n| n.to_str()).unwrap_or("unknown.wav");

        println!("   🔄 [{}/{}] {}", file_idx + 1, wav_files.len(), file_name);

        match load_audio_file(wav_file) {
            Ok((audio, sample_rate)) => {
                total_frames_processed += audio.len();

                match segment_vocalization_aggressive(file_name, file_idx, &audio, sample_rate) {
                    Ok(mut segments) => {
                        all_segments.append(&mut segments);
                    }
                    Err(e) => {
                        println!("        ⚠ Segmentation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("        ⚠ Failed to load audio: {}", e);
            }
        }
    }

    let level1_time = level1_start.elapsed();

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Level 1 Complete                                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("   ✅ Processed {} vocalizations", wav_files.len());
    println!("   📊 Total audio samples: {}", total_frames_processed);
    println!("   📝 Total phrase segments discovered: {}", all_segments.len());
    println!(
        "   ⏱️  Level 1 time: {:.2}s ({:.2}s per file)",
        level1_time.as_secs_f64(),
        level1_time.as_secs_f64() / wav_files.len() as f64
    );
    println!();

    if all_segments.is_empty() {
        println!("   ❌ No phrase segments discovered!");
        return Ok(());
    }

    // Build vocabulary across all vocalizations
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

        // Sort by frequency
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

    // Save all segments
    let segments_path = results_dir.join("all_segments.json");
    let segments_json = serde_json::to_string_pretty(&all_segments)?;
    fs::write(&segments_path, segments_json)?;
    println!("   💾 All segments: {}", segments_path.display());

    // Generate summary
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE 0 COMPLETE - TWO-LEVEL HDBSCAN                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  ✅ Two-level HDBSCAN for phrase discovery                                 ║");
    println!("║                                                                           ║");
    println!("║  📊 SUMMARY:                                                              ║");
    println!(
        "║     • Vocalizations processed: {}                                      ║",
        wav_files.len()
    );
    println!(
        "║     • Phrase segments discovered: {}                                  ║",
        all_segments.len()
    );
    println!(
        "║     • Vocabulary items discovered: {}                                  ║",
        vocabulary.len()
    );
    println!("║                                                                           ║");
    println!("║  🎯 NEXT STEPS:                                                           ║");
    println!("║     • Analyze vocabulary item distributions                             ║");
    println!("║     • Extract example audio for each vocabulary item                     ║");
    println!("║     • Build phrase transition statistics                                ║");
    println!("║     • Discover syntax rules                                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
