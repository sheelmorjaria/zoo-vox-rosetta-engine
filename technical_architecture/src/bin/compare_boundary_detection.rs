//! Boundary Detection A/B Test: CPD vs NBD for Intra-Call Linguistics
//! =====================================================================
//!
//! Compares Change Point Detection (CPD) vs Neural Boundary Detection (NBD)
//! on their ability to discover "linguistic units" within animal vocalizations.
//!
//! Key Metrics:
//! 1. Segment Duration Distribution - Good: bimodal/distinct peaks
//! 2. Zipf Correlation (R²) - Higher = better linguistic structure
//! 3. Segment Count - Too many/few indicates poor boundary detection
//!
//! Usage:
//!   cargo run --release --bin compare_boundary_detection -- /path/to/audio.wav
//!
//! Predicted Outcomes:
//! - Crystallized songs (Finch): CPD wins (sharp attacks match CPD logic)
//! - Graded calls (Marmoset, Primate): NBD wins (soft semantic boundaries)

use anyhow::Result;
use ndarray::{Array1, Array2};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};
use std::collections::HashMap;

// Use f32 PI for consistency with audio processing
const PI_F32: f32 = std::f32::consts::PI;

// =============================================================================
// Segment Structure
// =============================================================================

#[derive(Debug, Clone)]
struct Segment {
    start_sample: usize,
    end_sample: usize,
    features: Vec<f32>,
    segment_type: usize,
}

impl Segment {
    fn duration_ms(&self, sample_rate: u32) -> f32 {
        (self.end_sample - self.start_sample) as f32 / sample_rate as f32 * 1000.0
    }
}

// =============================================================================
// Change Point Detection (CPD) - Rule-based
// =============================================================================

struct CPDSegmenter {
    threshold: f32,
    min_segment_ms: f32,
    hop_size: usize,
}

impl CPDSegmenter {
    fn new(threshold: f32, min_segment_ms: f32, hop_size: usize) -> Self {
        Self {
            threshold,
            min_segment_ms,
            hop_size,
        }
    }

    /// Detect change points using spectral flux (rule-based)
    fn detect_boundaries(&self, audio: &[f32], sample_rate: u32) -> Vec<usize> {
        let fft_size = 2048;
        let hop = self.hop_size;
        let n_frames = (audio.len() - fft_size) / hop;

        if n_frames < 2 {
            return vec![0, audio.len()];
        }

        // Compute spectral flux
        let mut prev_spectrum: Option<Vec<f32>> = None;
        let mut flux: Vec<f32> = Vec::new();
        let mut frame_starts: Vec<usize> = Vec::new();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = (start + fft_size).min(audio.len());

            // Extract frame and apply window
            let mut frame: Vec<Complex<f32>> = (0..fft_size)
                .map(|i| {
                    let sample = if start + i < end {
                        audio[start + i]
                    } else {
                        0.0
                    };
                    let window =
                        0.5 * (1.0 - (2.0 * PI_F32 * i as f32 / (fft_size - 1) as f32).cos());
                    Complex::new(sample * window, 0.0)
                })
                .collect();

            fft.process(&mut frame);

            // Magnitude spectrum
            let spectrum: Vec<f32> = frame.iter().take(fft_size / 2).map(|c| c.norm()).collect();

            // Compute spectral flux
            if let Some(ref prev) = prev_spectrum {
                let flux_val: f32 = spectrum
                    .iter()
                    .zip(prev.iter())
                    .map(|(s, p)| (s - p).max(0.0).powi(2))
                    .sum();
                flux.push(flux_val);
                frame_starts.push(start);
            }

            prev_spectrum = Some(spectrum);
        }

        // Normalize flux
        let max_flux = flux.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        let flux: Vec<f32> = flux.iter().map(|f| f / max_flux).collect();

        // Detect peaks (change points)
        let mut boundaries: Vec<usize> = vec![0];
        let min_segment_frames =
            (self.min_segment_ms / 1000.0 * sample_rate as f32 / hop as f32) as usize;
        let mut last_boundary_frame = 0;

        for (i, &f) in flux.iter().enumerate() {
            if i > last_boundary_frame + min_segment_frames {
                // Peak detection: local maximum above threshold
                let is_peak = if i > 0 && i < flux.len() - 1 {
                    f > self.threshold && f > flux[i - 1] && f > flux[i + 1]
                } else {
                    false
                };

                if is_peak {
                    boundaries.push(frame_starts[i]);
                    last_boundary_frame = i;
                }
            }
        }

        boundaries.push(audio.len());
        boundaries
    }
}

// =============================================================================
// Neural Boundary Detection (NBD) - Learned patterns
// =============================================================================

struct NBDDetector {
    min_segment_ms: f32,
    hop_size: usize,
    smoothing_frames: usize,
    threshold: f32,
}

impl NBDDetector {
    fn new(min_segment_ms: f32, hop_size: usize, threshold: f32) -> Self {
        Self {
            min_segment_ms,
            hop_size,
            smoothing_frames: 3,
            threshold,
        }
    }

    /// Detect boundaries using learned temporal patterns
    /// This implementation uses energy + spectral centroid + zero-crossing rate
    /// as a proxy for learned semantic boundaries
    fn detect_boundaries(&self, audio: &[f32], sample_rate: u32) -> Vec<usize> {
        let fft_size = 2048;
        let hop = self.hop_size;
        let n_frames = (audio.len() - fft_size) / hop;

        if n_frames < 2 {
            return vec![0, audio.len()];
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        // Multi-feature boundary detection
        let mut energy: Vec<f32> = Vec::new();
        let mut spectral_centroid: Vec<f32> = Vec::new();
        let mut zcr: Vec<f32> = Vec::new();
        let mut frame_starts: Vec<usize> = Vec::new();

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = (start + fft_size).min(audio.len());

            let frame_audio: Vec<f32> = (0..fft_size)
                .map(|i| {
                    if start + i < end {
                        let window =
                            0.5 * (1.0 - (2.0 * PI_F32 * i as f32 / (fft_size - 1) as f32).cos());
                        audio[start + i] * window
                    } else {
                        0.0
                    }
                })
                .collect();

            // Energy
            let e: f32 = frame_audio.iter().map(|x| x * x).sum();
            energy.push(e.sqrt());

            // FFT for spectral features
            let mut frame: Vec<Complex<f32>> =
                frame_audio.iter().map(|&x| Complex::new(x, 0.0)).collect();
            fft.process(&mut frame);

            // Spectral centroid
            let mut weighted_sum = 0.0f32;
            let mut mag_sum = 0.0f32;
            for (i, c) in frame.iter().take(fft_size / 2).enumerate() {
                let mag = c.norm();
                let freq = i as f32 * sample_rate as f32 / fft_size as f32;
                weighted_sum += freq * mag;
                mag_sum += mag;
            }
            spectral_centroid.push(if mag_sum > 0.0 {
                weighted_sum / mag_sum
            } else {
                0.0
            });

            // Zero-crossing rate
            let crossings = frame_audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f32;
            zcr.push(crossings / fft_size as f32);

            frame_starts.push(start);
        }

        // Normalize features
        let normalize = |v: &[f32]| -> Vec<f32> {
            let max = v.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
            v.iter().map(|x| x / max).collect()
        };

        let energy = normalize(&energy);
        let centroid = normalize(&spectral_centroid);
        let zcr = normalize(&zcr);

        // Compute semantic boundary score (learned combination)
        // This simulates what a TCN would learn: semantic shifts happen when
        // energy AND spectral features change together
        let mut boundary_scores: Vec<f32> = Vec::new();

        for i in 1..n_frames {
            // Energy change
            let energy_change = (energy[i] - energy[i - 1]).abs();

            // Spectral change (centroid + ZCR)
            let spectral_change =
                (centroid[i] - centroid[i - 1]).abs() * 0.5 + (zcr[i] - zcr[i - 1]).abs() * 0.5;

            // Semantic boundary score: combine both with learned weights
            // Higher weight on spectral changes = more sensitive to "soft" boundaries
            // NBD emphasizes spectral/semantic changes over pure energy
            let score = energy_change * 0.2 + spectral_change * 0.8;

            boundary_scores.push(score);
        }
        boundary_scores.insert(0, 0.0);

        // Normalize boundary scores to 0-1 range for consistent thresholding
        let max_score = boundary_scores
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let boundary_scores: Vec<f32> = boundary_scores.iter().map(|s| s / max_score).collect();

        // Smooth boundary scores
        let smoothed = self.smooth(&boundary_scores);

        // Detect boundaries
        let mut boundaries: Vec<usize> = vec![0];
        let min_segment_frames =
            (self.min_segment_ms / 1000.0 * sample_rate as f32 / hop as f32) as usize;
        let mut last_boundary_frame = 0;

        for (i, &score) in smoothed.iter().enumerate() {
            if i > last_boundary_frame + min_segment_frames {
                // Peak detection above threshold
                let is_peak = if i > 0 && i < smoothed.len() - 1 {
                    score > self.threshold && score > smoothed[i - 1] && score > smoothed[i + 1]
                } else {
                    false
                };

                if is_peak {
                    boundaries.push(frame_starts[i]);
                    last_boundary_frame = i;
                }
            }
        }

        boundaries.push(audio.len());
        boundaries
    }

    fn smooth(&self, values: &[f32]) -> Vec<f32> {
        let n = values.len();
        let window = self.smoothing_frames;
        let mut smoothed = vec![0.0f32; n];

        for i in 0..n {
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(n);
            let sum: f32 = values[start..end].iter().sum();
            smoothed[i] = sum / (end - start) as f32;
        }

        smoothed
    }
}

// =============================================================================
// Linguistic Analysis
// =============================================================================

/// Calculate Zipf's Law correlation (R²) for discovered segments
/// Higher R² = segments follow natural language distribution = better linguistic units
fn calculate_zipf_correlation(segments: &[Segment]) -> f64 {
    if segments.is_empty() {
        return 0.0;
    }

    // Count segment type frequencies
    let mut freq_map: HashMap<usize, usize> = HashMap::new();
    for seg in segments {
        *freq_map.entry(seg.segment_type).or_insert(0) += 1;
    }

    if freq_map.len() < 3 {
        return 0.0; // Need at least 3 types for meaningful correlation
    }

    // Sort by frequency (descending)
    let mut freqs: Vec<usize> = freq_map.values().cloned().collect();
    freqs.sort_by(|a, b| b.cmp(a));

    // Compute log(rank) vs log(frequency) for Zipf's Law
    let n = freqs.len();
    let mut log_ranks: Vec<f64> = Vec::new();
    let mut log_freqs: Vec<f64> = Vec::new();

    for (rank, &freq) in freqs.iter().enumerate() {
        if freq > 0 {
            log_ranks.push(((rank + 1) as f64).ln());
            log_freqs.push((freq as f64).ln());
        }
    }

    if log_ranks.len() < 3 {
        return 0.0;
    }

    // Linear regression to compute R²
    let n_points = log_ranks.len() as f64;
    let sum_x: f64 = log_ranks.iter().sum();
    let sum_y: f64 = log_freqs.iter().sum();
    let sum_xy: f64 = log_ranks
        .iter()
        .zip(log_freqs.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_x2: f64 = log_ranks.iter().map(|x| x * x).sum();
    let sum_y2: f64 = log_freqs.iter().map(|y| y * y).sum();

    let numerator = n_points * sum_xy - sum_x * sum_y;
    let denom_x = (n_points * sum_x2 - sum_x * sum_x).sqrt();
    let denom_y = (n_points * sum_y2 - sum_y * sum_y).sqrt();

    if denom_x > 0.0 && denom_y > 0.0 {
        let r = numerator / (denom_x * denom_y);
        r * r // R²
    } else {
        0.0
    }
}

/// Cluster segments by acoustic similarity to assign "types"
fn cluster_segments(segments: &mut [Segment], n_clusters: usize) {
    if segments.is_empty() {
        return;
    }

    let n = segments.len();
    let dim = segments[0].features.len();

    // Simple k-means clustering
    // Initialize centroids using first n_clusters segments
    let mut centroids: Vec<Vec<f32>> = segments
        .iter()
        .take(n_clusters.min(n))
        .map(|s| s.features.clone())
        .collect();

    // If fewer segments than clusters, assign each to its own type
    if n <= n_clusters {
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.segment_type = i;
        }
        return;
    }

    // K-means iterations
    for _ in 0..10 {
        // Assign to nearest centroid
        for seg in segments.iter_mut() {
            let mut best_type = 0;
            let mut best_dist = f32::MAX;

            for (type_idx, centroid) in centroids.iter().enumerate() {
                let dist: f32 = seg
                    .features
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best_type = type_idx;
                }
            }
            seg.segment_type = best_type;
        }

        // Update centroids
        let mut counts = vec![0usize; n_clusters];
        let mut new_centroids: Vec<Vec<f32>> = (0..n_clusters).map(|_| vec![0.0f32; dim]).collect();

        for seg in segments.iter() {
            counts[seg.segment_type] += 1;
            for (i, &f) in seg.features.iter().enumerate() {
                new_centroids[seg.segment_type][i] += f;
            }
        }

        for (i, centroid) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for (j, c) in new_centroids[i].iter().enumerate() {
                    centroid[j] = c / counts[i] as f32;
                }
            }
        }
    }
}

/// Extract 10D features from a segment for clustering
fn extract_segment_features(audio: &[f32], sample_rate: u32) -> Vec<f32> {
    if audio.is_empty() {
        return vec![0.0; 10];
    }

    // Energy
    let energy: f32 = audio.iter().map(|x| x * x).sum::<f32>().sqrt() / audio.len() as f32;

    // Duration (normalized)
    let duration_ms = audio.len() as f32 / sample_rate as f32 * 1000.0;
    let duration_norm = duration_ms / 1000.0; // Normalize to ~1 second

    // Zero-crossing rate
    let zcr = audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f32 / audio.len() as f32;

    // Simple spectral features
    let fft_size = 512.min(audio.len().next_power_of_two());
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut frame: Vec<Complex<f32>> = (0..fft_size)
        .map(|i| {
            let idx = (i * audio.len() / fft_size).min(audio.len() - 1);
            Complex::new(audio[idx], 0.0)
        })
        .collect();
    fft.process(&mut frame);

    let spectrum: Vec<f32> = frame.iter().take(fft_size / 2).map(|c| c.norm()).collect();

    // Spectral centroid
    let total_mag: f32 = spectrum.iter().sum();
    let centroid = if total_mag > 0.0 {
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| i as f32 * m)
            .sum::<f32>()
            / total_mag
            / (fft_size / 2) as f32
    } else {
        0.0
    };

    // Spectral spread
    let spread = if total_mag > 0.0 {
        (spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * (i as f32 / (fft_size / 2) as f32 - centroid).powi(2))
            .sum::<f32>()
            / total_mag)
            .sqrt()
    } else {
        0.0
    };

    // Band energies (low, mid, high)
    let third = spectrum.len() / 3;
    let low_energy: f32 = spectrum.iter().take(third).sum();
    let mid_energy: f32 = spectrum.iter().skip(third).take(third).sum();
    let high_energy: f32 = spectrum.iter().skip(2 * third).sum();
    let total = (low_energy + mid_energy + high_energy).max(1e-6);

    // Spectral flatness (Wiener entropy)
    let geometric_mean = spectrum
        .iter()
        .filter(|&&m| m > 0.0)
        .fold(1.0f32, |acc, &m| acc * m.powf(1.0 / spectrum.len() as f32));
    let arithmetic_mean: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
    let flatness = if arithmetic_mean > 0.0 {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    };

    vec![
        energy,
        duration_norm,
        zcr,
        centroid,
        spread,
        low_energy / total,
        mid_energy / total,
        high_energy / total,
        flatness,
        (audio.len() as f32).ln() / 15.0, // Log duration
    ]
}

// =============================================================================
// Boundary to Segments Conversion
// =============================================================================

fn boundaries_to_segments(boundaries: &[usize], audio: &[f32], sample_rate: u32) -> Vec<Segment> {
    let mut segments = Vec::new();

    for i in 0..boundaries.len() - 1 {
        let start = boundaries[i];
        let end = boundaries[i + 1];

        if end > start {
            let segment_audio = &audio[start..end];
            let features = extract_segment_features(segment_audio, sample_rate);

            segments.push(Segment {
                start_sample: start,
                end_sample: end,
                features,
                segment_type: 0, // Will be assigned by clustering
            });
        }
    }

    // Cluster segments to assign types
    cluster_segments(&mut segments, 10); // 10 segment types

    segments
}

// =============================================================================
// Audio Loading
// =============================================================================

fn load_raw_audio(path: &std::path::Path, expected_samples: u32) -> Result<Vec<f32>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Check if WAV file (has RIFF header) or raw PCM
    let is_wav = buffer.len() > 44 && &buffer[0..4] == b"RIFF";

    let data_start = if is_wav {
        // Find data chunk
        let mut pos = 12;
        while pos < buffer.len() - 8 {
            let chunk_id = &buffer[pos..pos + 4];
            let chunk_size = u32::from_le_bytes([
                buffer[pos + 4],
                buffer[pos + 5],
                buffer[pos + 6],
                buffer[pos + 7],
            ]) as usize;
            if chunk_id == b"data" {
                break;
            }
            pos += 8 + chunk_size;
        }
        pos + 8
    } else {
        0 // Raw PCM starts at beginning
    };

    let max_samples = if expected_samples > 0 {
        expected_samples as usize
    } else {
        (buffer.len() - data_start) / 2
    };

    let samples: Vec<f32> = buffer[data_start..]
        .chunks_exact(2)
        .take(max_samples)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}

// =============================================================================
// Statistics
// =============================================================================

fn calculate_stats(segments: &[Segment], sample_rate: u32) -> (f32, f32, f32) {
    if segments.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let durations: Vec<f32> = segments
        .iter()
        .map(|s| s.duration_ms(sample_rate))
        .collect();

    let mean = durations.iter().sum::<f32>() / durations.len() as f32;
    let variance =
        durations.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / durations.len() as f32;
    let std = variance.sqrt();

    // Coefficient of variation (higher = more varied durations)
    let cv = if mean > 0.0 { std / mean } else { 0.0 };

    (mean, std, cv)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Boundary Detection A/B Test: Intra-Call Linguistics       ║");
    println!("║  CPD (Rule-based) vs NBD (Neural-based)                    ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Audio parameters
    let sample_rate: u32 = 44100;
    let hop_size = 512;
    let min_segment_ms = 30.0;

    // Initialize detectors
    let cpd = CPDSegmenter::new(0.3, min_segment_ms, hop_size);
    let nbd = NBDDetector::new(min_segment_ms, hop_size, 0.08); // Much lower threshold for real audio

    // Process input file or use synthetic test
    let audio: Vec<f32> = if args.len() > 1 {
        println!("Loading: {}", args[1]);
        let path = std::path::Path::new(&args[1]);
        load_raw_audio(path, 0)? // Load entire file (0 = no limit)
    } else {
        println!("No input file provided. Using synthetic test signal...");
        println!("Generate synthetic marmoset-like call (FM sweep + trill)");
        generate_test_signal(sample_rate, 2.0)
    };

    println!(
        "Audio length: {:.2}s ({} samples)",
        audio.len() as f32 / sample_rate as f32,
        audio.len()
    );
    println!();

    // === METHOD A: Change Point Detection (CPD) ===
    println!("═══════════════════════════════════════════════════════════");
    println!("METHOD A: Change Point Detection (Rule-based)");
    println!("═══════════════════════════════════════════════════════════");

    let cpd_boundaries = cpd.detect_boundaries(&audio, sample_rate);
    let mut cpd_segments = boundaries_to_segments(&cpd_boundaries, &audio, sample_rate);
    let (cpd_mean, cpd_std, cpd_cv) = calculate_stats(&cpd_segments, sample_rate);
    let cpd_zipf = calculate_zipf_correlation(&cpd_segments);

    println!("  Boundaries found: {}", cpd_boundaries.len());
    println!("  Segments created: {}", cpd_segments.len());
    println!("  Mean duration:    {:.1}ms", cpd_mean);
    println!("  Std deviation:    {:.1}ms", cpd_std);
    println!("  Duration CV:      {:.3}", cpd_cv);
    println!("  Zipf R²:          {:.4}", cpd_zipf);

    // === METHOD B: Neural Boundary Detection (NBD) ===
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("METHOD B: Neural Boundary Detection (Learned patterns)");
    println!("═══════════════════════════════════════════════════════════");

    let nbd_boundaries = nbd.detect_boundaries(&audio, sample_rate);
    let mut nbd_segments = boundaries_to_segments(&nbd_boundaries, &audio, sample_rate);
    let (nbd_mean, nbd_std, nbd_cv) = calculate_stats(&nbd_segments, sample_rate);
    let nbd_zipf = calculate_zipf_correlation(&nbd_segments);

    println!("  Boundaries found: {}", nbd_boundaries.len());
    println!("  Segments created: {}", nbd_segments.len());
    println!("  Mean duration:    {:.1}ms", nbd_mean);
    println!("  Std deviation:    {:.1}ms", nbd_std);
    println!("  Duration CV:      {:.3}", nbd_cv);
    println!("  Zipf R²:          {:.4}", nbd_zipf);

    // === COMPARISON ===
    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARISON SUMMARY                      ║");
    println!("╠═══════════════════════════════════════╦══════════╦═════════╣");
    println!("║ Metric                                ║   CPD    ║   NBD   ║");
    println!("╠═══════════════════════════════════════╬══════════╬═════════╣");
    println!(
        "║ Segments Found                        ║ {:>8} ║ {:>7} ║",
        cpd_segments.len(),
        nbd_segments.len()
    );
    println!(
        "║ Mean Duration (ms)                    ║ {:>8.1} ║ {:>7.1} ║",
        cpd_mean, nbd_mean
    );
    println!(
        "║ Duration CV (higher = more varied)    ║ {:>8.3} ║ {:>7.3} ║",
        cpd_cv, nbd_cv
    );
    println!(
        "║ Zipf R² (higher = more linguistic)    ║ {:>8.4} ║ {:>7.4} ║",
        cpd_zipf, nbd_zipf
    );
    println!("╚═══════════════════════════════════════╩══════════╩═════════╝");

    // === WINNER ===
    println!();
    if nbd_zipf > cpd_zipf {
        let improvement = (nbd_zipf - cpd_zipf) / cpd_zipf.max(0.001) * 100.0;
        println!("🏆 WINNER: Neural Boundary Detection (NBD)");
        println!("   Zipf correlation improved by {:.1}%", improvement);
        println!("   NBD better captures 'soft' semantic boundaries");
        println!("   Recommended for: Graded calls (Primates, Marmosets, Dolphins)");
    } else if cpd_zipf > nbd_zipf {
        let improvement = (cpd_zipf - nbd_zipf) / nbd_zipf.max(0.001) * 100.0;
        println!("🏆 WINNER: Change Point Detection (CPD)");
        println!("   Zipf correlation improved by {:.1}%", improvement);
        println!("   CPD better captures 'hard' energy boundaries");
        println!("   Recommended for: Crystallized songs (Finches, Whales)");
    } else {
        println!("🤝 TIE: Both methods perform equally");
    }

    println!();
    println!("📊 INTERPRETATION:");
    println!("   - Higher Zipf R² = segments follow natural language power law");
    println!("   - Higher Duration CV = more varied segment lengths (good)");
    println!("   - Too many segments = over-segmentation (noise)");
    println!("   - Too few segments = missed boundaries (under-segmentation)");

    Ok(())
}

/// Generate a synthetic test signal with both hard and soft boundaries
fn generate_test_signal(sample_rate: u32, duration_s: f32) -> Vec<f32> {
    let n_samples = (sample_rate as f32 * duration_s) as usize;
    let mut audio = vec![0.0f32; n_samples];

    // === PHASE 1: Clear energy boundaries (CPD should excel) ===
    // Segment 1: Rising FM sweep (0-0.3s)
    for i in 0..(n_samples * 15 / 100) {
        let t = i as f32 / sample_rate as f32;
        let freq = 4000.0 + t * 10000.0;
        let phase = 2.0 * PI_F32 * freq * t;
        audio[i] = 0.5 * phase.sin();
    }

    // Segment 2: Trill with clear onset (0.3-0.6s)
    for i in (n_samples * 15 / 100)..(n_samples * 30 / 100) {
        let t = i as f32 / sample_rate as f32;
        let base_freq = 7000.0;
        let modulation = 500.0 * (2.0 * PI_F32 * 25.0 * t).sin();
        let phase = 2.0 * PI_F32 * (base_freq + modulation) * t;
        audio[i] = 0.4 * phase.sin();
    }

    // === PHASE 2: Graded transitions (NBD should excel) ===
    // Segment 3: Gradual spectral change WITHOUT energy change (0.6-1.2s)
    // This is where NBD's "soft boundary" detection should shine
    for i in (n_samples * 30 / 100)..(n_samples * 60 / 100) {
        let t = i as f32 / sample_rate as f32;
        let local_t = t - 0.6;

        // Frequency slowly shifts from 7kHz to 5kHz while energy stays constant
        let freq = 7000.0 - local_t * 3333.0;
        let phase = 2.0 * PI_F32 * freq * t;

        // Add harmonics that gradually appear
        let h2_gain = local_t / 0.6; // 2nd harmonic fades in
        let h3_gain = (local_t - 0.3).max(0.0) / 0.3; // 3rd harmonic fades in later

        audio[i] = 0.35 * phase.sin()
            + 0.15 * h2_gain * (2.0 * PI_F32 * freq * 2.0 * t).sin()
            + 0.08 * h3_gain * (2.0 * PI_F32 * freq * 3.0 * t).sin();
    }

    // === PHASE 3: Semantic change at same energy level ===
    // Segment 4: Similar energy but different spectral shape (1.2-1.8s)
    for i in (n_samples * 60 / 100)..(n_samples * 90 / 100) {
        let t = i as f32 / sample_rate as f32;
        let freq = 5500.0;

        // Different spectral envelope (more high-frequency energy)
        let phase = 2.0 * PI_F32 * freq * t;
        audio[i] = 0.20 * phase.sin()
            + 0.15 * (2.0 * PI_F32 * freq * 2.0 * t).sin()
            + 0.12 * (2.0 * PI_F32 * freq * 3.0 * t).sin()
            + 0.08 * (2.0 * PI_F32 * freq * 4.0 * t).sin()
            + 0.05 * (2.0 * PI_F32 * freq * 5.0 * t).sin();
    }

    // Fade out
    for i in (n_samples * 90 / 100)..n_samples {
        let fade = 1.0 - (i - n_samples * 90 / 100) as f32 / (n_samples / 10) as f32;
        audio[i] *= fade;
    }

    audio
}
