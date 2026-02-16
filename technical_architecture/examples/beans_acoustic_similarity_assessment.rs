// BEANS-Zero Acoustic Similarity Assessment with Chunked Processing
//
// Processes the entire BEANS-Zero dataset using:
// 1. Chunked feature extraction (memory-efficient)
// 2. Acoustic Similarity Engine (no O(n²) distance matrix)
// 3. Streaming k-NN classification
// 4. Progressive type discovery
//
// Usage:
//   cargo run --release --example beans_acoustic_similarity_assessment
//
// This approach avoids the O(n²) memory issue by using similarity-based
// grouping instead of full pairwise distance matrices.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use ndarray::Array2;

// Import acoustic similarity engine from the library
// Note: Library re-exports DistanceMetric as SimilarityMetric
use technical_architecture::{
    AcousticSimilarityEngine, SimilarityMetric,
};

// ============================================================================
// Configuration
// ============================================================================

const CHUNK_SIZE: usize = 5000;        // Process 5000 samples per chunk
const SIMILARITY_THRESHOLD: f64 = 0.85; // For type grouping
const FEATURE_DIM: usize = 30;
const K_NEIGHBORS: usize = 10;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    dataset: String,
    split: String,
    samples: Vec<SampleInfo>,
    resample_rate: u32,
    label_columns: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SampleInfo {
    id: String,
    audio_file: String,
    n_samples: usize,
    duration_ms: f64,
    labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct ExtractedFeatures {
    sample_id: String,
    features: Vec<f64>,
    duration_ms: f64,
    labels: HashMap<String, String>,
    extraction_time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ChunkResults {
    chunk_id: usize,
    n_samples: usize,
    n_types: usize,
    type_distribution: Vec<(String, usize)>,
    avg_intra_similarity: f64,
    extraction_time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AcousticType {
    type_id: String,
    centroid: Vec<f64>,
    count: usize,
    sample_ids: Vec<String>,
    avg_distance_to_centroid: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GlobalAssessment {
    dataset: String,
    total_samples: usize,
    total_chunks: usize,
    feature_dim: usize,

    // Performance
    total_time_sec: f64,
    throughput_samples_per_sec: f64,

    // Type Discovery
    global_types: usize,
    type_entropy: f64,

    // Classification
    knn_accuracy: f64,
    knn_best_k: usize,

    // Similarity Statistics
    avg_intra_type_similarity: f64,
    avg_inter_type_distance: f64,
    separation_ratio: f64,

    // Label Analysis
    source_datasets: HashMap<String, usize>,
    task_types: HashMap<String, usize>,
}

// ============================================================================
// Fast Feature Extractor (from original example)
// ============================================================================

struct FastFeatureExtractor {
    sample_rate: u32,
    feature_dim: usize,
}

impl FastFeatureExtractor {
    fn new(sample_rate: u32, feature_dim: usize) -> Self {
        Self { sample_rate, feature_dim }
    }

    fn extract(&self, audio: &[f32]) -> Result<Vec<f64>> {
        let mut features = vec![0.0; self.feature_dim];

        if audio.is_empty() {
            return Ok(features);
        }

        let n = audio.len();
        let duration_ms = n as f64 / self.sample_rate as f64 * 1000.0;

        // Fundamental features
        features[0] = self.estimate_f0(audio).unwrap_or(0.0);
        features[1] = duration_ms;
        features[2] = self.estimate_f0_range(audio).unwrap_or(0.0);

        // Energy features
        let rms = (audio.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64).sqrt();
        let energy = audio.iter().map(|x| x.abs() as f64).sum::<f64>() / n as f64;
        features[3] = rms;
        features[4] = energy;

        // Spectral features (simplified)
        let spectrum = self.compute_spectrum(audio);
        features[5] = self.compute_hnr(&spectrum);
        features[6] = self.compute_flatness(&spectrum);
        features[7] = self.compute_harmonicity(&spectrum);

        // Temporal features
        let (attack, decay, sustain, release) = self.compute_envelope(audio);
        features[8] = attack;
        features[9] = decay;
        features[10] = sustain;
        features[11] = release;

        // Spectral centroid
        features[12] = self.compute_centroid(&spectrum);

        // Modulation features
        let (vib_rate, vib_depth) = self.estimate_vibrato(audio);
        features[13] = vib_rate;
        features[14] = vib_depth;

        // Perturbation features
        features[15] = self.compute_jitter(audio);
        features[16] = self.compute_shimmer(audio);

        // Spectral contrast (simplified MFCC-like)
        for i in 0..10 {
            if 17 + i < self.feature_dim {
                features[17 + i] = self.compute_spectral_band(&spectrum, i);
            }
        }

        // Rhythm features
        let (ici, onset_rate, ici_cv) = self.compute_rhythm(audio);
        if 27 < self.feature_dim { features[27] = ici; }
        if 28 < self.feature_dim { features[28] = onset_rate; }
        if 29 < self.feature_dim { features[29] = ici_cv; }

        Ok(features)
    }

    fn estimate_f0(&self, audio: &[f32]) -> Option<f64> {
        let n = audio.len();
        if n < 100 { return None; }

        // Autocorrelation-based F0 estimation
        let mut best_lag = 0;
        let mut best_corr = 0.0f64;
        let min_lag = (self.sample_rate as f64 / 20000.0) as usize;
        let max_lag = (self.sample_rate as f64 / 100.0).min(n as f64 / 2.0) as usize;

        let mean = audio.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let variance: f64 = audio.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        if variance < 1e-10 { return None; }

        for lag in min_lag..max_lag {
            let mut corr = 0.0;
            for i in 0..(n - lag) {
                corr += (audio[i] as f64 - mean) * (audio[i + lag] as f64 - mean);
            }
            corr /= (n - lag) as f64 * variance;

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        if best_lag > 0 && best_corr > 0.3 {
            Some(self.sample_rate as f64 / best_lag as f64)
        } else {
            None
        }
    }

    fn estimate_f0_range(&self, audio: &[f32]) -> Option<f64> {
        // Estimate F0 variation over time
        let window_size = (self.sample_rate as f64 * 0.05) as usize;
        let mut f0s = Vec::new();

        for start in (0..audio.len()).step_by(window_size) {
            let end = (start + window_size).min(audio.len());
            let window = &audio[start..end];
            if let Some(f0) = self.estimate_f0(window) {
                if f0 > 50.0 && f0 < 20000.0 {
                    f0s.push(f0);
                }
            }
        }

        if f0s.len() < 2 {
            return None;
        }

        let min_f0 = f0s.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_f0 = f0s.iter().cloned().fold(0.0f64, f64::max);

        Some(max_f0 - min_f0)
    }

    fn compute_spectrum(&self, audio: &[f32]) -> Vec<f64> {
        let n = audio.len().next_power_of_two();
        let mut spectrum = vec![0.0f64; n / 2];

        // Simple DFT for magnitude spectrum (simplified)
        for k in 0..n/2 {
            let mut real = 0.0;
            let mut imag = 0.0;
            for t in 0..audio.len().min(n) {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                real += audio[t] as f64 * angle.cos();
                imag += audio[t] as f64 * angle.sin();
            }
            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        spectrum
    }

    fn compute_hnr(&self, spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() { return 0.0; }
        let total: f64 = spectrum.iter().sum();
        if total == 0.0 { return 0.0; }

        // Simplified HNR based on spectral peaks
        let mut peaks = 0;
        for i in 1..spectrum.len()-1 {
            if spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1] {
                peaks += 1;
            }
        }
        (peaks as f64 / spectrum.len() as f64 * 30.0).min(30.0)
    }

    fn compute_flatness(&self, spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() { return 0.0; }

        let nonzero: Vec<f64> = spectrum.iter().cloned().filter(|&x| x > 1e-10).collect();
        if nonzero.is_empty() { return 0.0; }

        let geometric_mean = nonzero.iter().product::<f64>().powf(1.0 / nonzero.len() as f64);
        let arithmetic_mean = nonzero.iter().sum::<f64>() / nonzero.len() as f64;

        if arithmetic_mean > 0.0 {
            (geometric_mean / arithmetic_mean).min(1.0).max(0.0)
        } else {
            0.0
        }
    }

    fn compute_harmonicity(&self, spectrum: &[f64]) -> f64 {
        // Simplified harmonicity based on spectral periodicity
        if spectrum.len() < 10 { return 0.0; }

        let mut autocorr = 0.0;
        let mut energy = 0.0;

        for i in 0..spectrum.len()-10 {
            energy += spectrum[i] * spectrum[i];
            for lag in 1..=10 {
                if i + lag < spectrum.len() {
                    autocorr += spectrum[i] * spectrum[i + lag];
                }
            }
        }

        if energy > 0.0 { (autocorr / energy).min(1.0).max(0.0) } else { 0.0 }
    }

    fn compute_envelope(&self, audio: &[f32]) -> (f64, f64, f64, f64) {
        let n = audio.len();
        if n < 10 { return (0.0, 0.0, 0.0, 0.0); }

        // Compute amplitude envelope
        let envelope: Vec<f64> = audio.iter().map(|&x| x.abs() as f64).collect();

        let max_amp = envelope.iter().cloned().fold(0.0f64, f64::max);
        if max_amp == 0.0 { return (0.0, 0.0, 0.0, 0.0); }

        let threshold_90 = max_amp * 0.9;
        let threshold_10 = max_amp * 0.1;
        let threshold_50 = max_amp * 0.5;

        // Attack: time from 10% to 90%
        let mut attack_end = 0;
        for (i, &amp) in envelope.iter().enumerate() {
            if amp >= threshold_90 {
                attack_end = i;
                break;
            }
        }
        let mut attack_start = 0;
        for (i, &amp) in envelope[..attack_end].iter().enumerate() {
            if amp >= threshold_10 {
                attack_start = i;
                break;
            }
        }
        let attack = (attack_end - attack_start) as f64 / self.sample_rate as f64 * 1000.0;

        // Decay: time from peak to sustain
        let peak_idx = envelope.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut decay_end = peak_idx;
        for i in peak_idx..n {
            if envelope[i] <= threshold_50 {
                decay_end = i;
                break;
            }
        }
        let decay = (decay_end - peak_idx) as f64 / self.sample_rate as f64 * 1000.0;

        // Sustain: average level in middle portion
        let sustain_start = (n as f64 * 0.3) as usize;
        let sustain_end = (n as f64 * 0.7) as usize;
        let sustain = if sustain_end > sustain_start {
            envelope[sustain_start..sustain_end].iter().sum::<f64>()
                / (sustain_end - sustain_start) as f64 / max_amp
        } else { 0.0 };

        // Release: time from sustain to end
        let release_start = (n as f64 * 0.7) as usize;
        let release = (n - release_start) as f64 / self.sample_rate as f64 * 1000.0;

        (attack, decay, sustain, release)
    }

    fn compute_centroid(&self, spectrum: &[f64]) -> f64 {
        if spectrum.is_empty() { return 0.0; }

        let weighted_sum: f64 = spectrum.iter().enumerate()
            .map(|(i, &mag)| i as f64 * mag)
            .sum();
        let total_mag: f64 = spectrum.iter().sum();

        if total_mag > 0.0 {
            weighted_sum / total_mag * self.sample_rate as f64 / spectrum.len() as f64 / 2.0
        } else {
            0.0
        }
    }

    fn estimate_vibrato(&self, audio: &[f32]) -> (f64, f64) {
        // Estimate vibrato from F0 contour
        let window_size = (self.sample_rate as f64 * 0.02) as usize;
        let mut f0_contour = Vec::new();

        for start in (0..audio.len()).step_by(window_size / 2) {
            let end = (start + window_size).min(audio.len());
            let window = &audio[start..end];
            if let Some(f0) = self.estimate_f0(window) {
                if f0 > 50.0 && f0 < 20000.0 {
                    f0_contour.push(f0);
                }
            }
        }

        if f0_contour.len() < 4 {
            return (0.0, 0.0);
        }

        // Detect oscillation in F0 contour
        let mean_f0 = f0_contour.iter().sum::<f64>() / f0_contour.len() as f64;
        let mut crossings = 0;
        let mut above = f0_contour[0] > mean_f0;

        for f0 in &f0_contour[1..] {
            let now_above = *f0 > mean_f0;
            if now_above != above {
                crossings += 1;
                above = now_above;
            }
        }

        // Vibrato rate (crossings per second)
        let duration = audio.len() as f64 / self.sample_rate as f64;
        let vib_rate = crossings as f64 / duration / 2.0;

        // Vibrato depth (standard deviation of F0)
        let variance: f64 = f0_contour.iter()
            .map(|f| (f - mean_f0).powi(2))
            .sum::<f64>() / f0_contour.len() as f64;
        let vib_depth = variance.sqrt();

        (vib_rate.min(50.0), vib_depth.min(1000.0))
    }

    fn compute_jitter(&self, audio: &[f32]) -> f64 {
        // Period-to-period variation
        let n = audio.len();
        if n < 100 { return 0.0; }

        // Find zero crossings
        let mut periods = Vec::new();
        let mut last_crossing = 0;

        for i in 1..n {
            if audio[i - 1] < 0.0 && audio[i] >= 0.0 {
                if last_crossing > 0 {
                    periods.push(i - last_crossing);
                }
                last_crossing = i;
            }
        }

        if periods.len() < 3 { return 0.0; }

        let mean_period = periods.iter().sum::<usize>() as f64 / periods.len() as f64;
        let variance: f64 = periods.iter()
            .map(|p| (*p as f64 - mean_period).powi(2))
            .sum::<f64>() / periods.len() as f64;

        (variance.sqrt() / mean_period).min(1.0).max(0.0)
    }

    fn compute_shimmer(&self, audio: &[f32]) -> f64 {
        // Amplitude variation between periods
        let n = audio.len();
        if n < 100 { return 0.0; }

        // Find peak amplitudes per period
        let mut peaks = Vec::new();
        let mut in_period = false;
        let mut max_in_period = 0.0f32;

        for i in 1..n {
            if audio[i - 1] < 0.0 && audio[i] >= 0.0 {
                if in_period && max_in_period > 0.0 {
                    peaks.push(max_in_period as f64);
                }
                in_period = true;
                max_in_period = 0.0;
            }
            if in_period {
                max_in_period = max_in_period.max(audio[i].abs());
            }
        }

        if peaks.len() < 3 { return 0.0; }

        let mean_peak = peaks.iter().sum::<f64>() / peaks.len() as f64;
        if mean_peak == 0.0 { return 0.0; }

        let variance: f64 = peaks.iter()
            .map(|p| (p - mean_peak).powi(2))
            .sum::<f64>() / peaks.len() as f64;

        (variance.sqrt() / mean_peak).min(1.0).max(0.0)
    }

    fn compute_spectral_band(&self, spectrum: &[f64], band_idx: usize) -> f64 {
        if spectrum.is_empty() { return 0.0; }

        let n_bands = 10;
        let band_size = spectrum.len() / n_bands;
        let start = band_idx * band_size;
        let end = if band_idx == n_bands - 1 { spectrum.len() } else { start + band_size };

        if end <= start || end > spectrum.len() { return 0.0; }

        spectrum[start..end].iter().sum::<f64>() / (end - start) as f64
    }

    fn compute_rhythm(&self, audio: &[f32]) -> (f64, f64, f64) {
        let n = audio.len();
        if n < 100 { return (0.0, 0.0, 0.0); }

        // Onset detection using energy derivative
        let window = (self.sample_rate as f64 * 0.01) as usize;
        let mut onsets = Vec::new();

        let mut prev_energy = 0.0;
        for start in (0..n).step_by(window) {
            let end = (start + window).min(n);
            let energy = audio[start..end].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();

            if energy > prev_energy * 2.0 && prev_energy > 0.0 {
                onsets.push(start);
            }
            prev_energy = energy;
        }

        if onsets.len() < 2 {
            return (0.0, 0.0, 0.0);
        }

        // Inter-onset intervals
        let mut icis: Vec<f64> = onsets.windows(2)
            .map(|w| (w[1] - w[0]) as f64 / self.sample_rate as f64 * 1000.0)
            .collect();

        let mean_ici = icis.iter().sum::<f64>() / icis.len() as f64;
        let variance = icis.iter().map(|ici| (ici - mean_ici).powi(2)).sum::<f64>() / icis.len() as f64;
        let std_ici = variance.sqrt();
        let ici_cv = if mean_ici > 0.0 { std_ici / mean_ici } else { 0.0 };

        let onset_rate = if mean_ici > 0.0 { 1000.0 / mean_ici } else { 0.0 };

        (mean_ici, onset_rate, ici_cv)
    }
}

// ============================================================================
// Chunked Processor with Acoustic Similarity
// ============================================================================

struct ChunkedSimilarityProcessor {
    chunk_size: usize,
    sample_rate: u32,
    feature_dim: usize,
    similarity_threshold: f64,
}

impl ChunkedSimilarityProcessor {
    fn new(chunk_size: usize, sample_rate: u32, feature_dim: usize, similarity_threshold: f64) -> Self {
        Self {
            chunk_size,
            sample_rate,
            feature_dim,
            similarity_threshold,
        }
    }

    /// Process all chunks and return features + type assignments
    fn process_all_chunks(
        &self,
        manifest: &Manifest,
        base_path: &Path,
    ) -> (Vec<ExtractedFeatures>, Vec<AcousticType>, Vec<ChunkResults>) {
        let n_samples = manifest.samples.len();
        let n_chunks = (n_samples + self.chunk_size - 1) / self.chunk_size;

        println!("Processing {} samples in {} chunks of {}...",
            n_samples, n_chunks, self.chunk_size);
        println!();

        let all_features = Arc::new(std::sync::Mutex::new(Vec::new()));
        let chunk_results = Arc::new(std::sync::Mutex::new(Vec::new()));
        let processed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));

        let start_time = Instant::now();

        // Process chunks
        for chunk_id in 0..n_chunks {
            let start_idx = chunk_id * self.chunk_size;
            let end_idx = (start_idx + self.chunk_size).min(n_samples);
            let chunk_samples = &manifest.samples[start_idx..end_idx];

            let chunk_start = Instant::now();
            let extractor = FastFeatureExtractor::new(self.sample_rate, self.feature_dim);

            let mut chunk_features = Vec::new();

            for sample in chunk_samples {
                let audio_path = base_path.join(&sample.audio_file);

                match self.load_raw_audio(&audio_path, sample.n_samples) {
                    Ok(audio) => {
                        let t0 = Instant::now();
                        match extractor.extract(&audio) {
                            Ok(features) => {
                                let extraction_time = t0.elapsed().as_secs_f64() * 1000.0;
                                processed.fetch_add(1, Ordering::Relaxed);

                                chunk_features.push(ExtractedFeatures {
                                    sample_id: sample.id.clone(),
                                    features,
                                    duration_ms: sample.duration_ms,
                                    labels: sample.labels.clone(),
                                    extraction_time_ms: extraction_time,
                                });
                            }
                            Err(_) => {
                                failed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            let chunk_time = chunk_start.elapsed().as_secs_f64() * 1000.0;

            // Discover types within chunk using acoustic similarity
            let (types, type_dist, avg_sim) = self.discover_chunk_types(&chunk_features);

            let result = ChunkResults {
                chunk_id,
                n_samples: chunk_features.len(),
                n_types: types.len(),
                type_distribution: type_dist,
                avg_intra_similarity: avg_sim,
                extraction_time_ms: chunk_time,
            };

            // Progress output
            println!("  Chunk {}/{}: {} samples, {} types, {:.2}s",
                chunk_id + 1, n_chunks,
                chunk_features.len(),
                types.len(),
                chunk_time / 1000.0);

            all_features.lock().unwrap().extend(chunk_features);
            chunk_results.lock().unwrap().push(result);
        }

        let elapsed = start_time.elapsed();
        let n_processed = processed.load(Ordering::Relaxed);
        let n_failed = failed.load(Ordering::Relaxed);
        let throughput = n_processed as f64 / elapsed.as_secs_f64();

        println!();
        println!("Extraction complete:");
        println!("  ├─ Processed: {} samples", n_processed);
        println!("  ├─ Failed: {} samples", n_failed);
        println!("  ├─ Time: {:.2}s", elapsed.as_secs_f64());
        println!("  └─ Throughput: {:.1} samples/sec", throughput);
        println!();

        let features = Arc::try_unwrap(all_features).unwrap().into_inner().unwrap();
        let results = Arc::try_unwrap(chunk_results).unwrap().into_inner().unwrap();

        // Build global types from all features
        let global_types = self.build_global_types(&features);

        (features, global_types, results)
    }

    fn discover_chunk_types(
        &self,
        features: &[ExtractedFeatures],
    ) -> (Vec<AcousticType>, Vec<(String, usize)>, f64) {
        if features.is_empty() {
            return (Vec::new(), Vec::new(), 0.0);
        }

        let n = features.len();

        // Create feature matrix for this chunk
        let mut matrix = Array2::<f64>::zeros((n, self.feature_dim));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(self.feature_dim) {
                matrix[[i, j]] = val;
            }
        }

        // Create similarity engine
        let mut engine = AcousticSimilarityEngine::with_metric(
            self.feature_dim,
            SimilarityMetric::Cosine,
        );
        engine.fit_normalization(&matrix);

        // Group by similarity
        let mut assigned = vec![false; n];
        let mut types: Vec<AcousticType> = Vec::new();
        let mut total_sim = 0.0;
        let mut sim_count = 0;

        for i in 0..n {
            if assigned[i] {
                continue;
            }

            let mut group_indices = vec![i];
            assigned[i] = true;

            let query = matrix.row(i).to_owned();

            for j in (i + 1)..n {
                if assigned[j] {
                    continue;
                }

                let candidate = matrix.row(j).to_owned();
                let sim = engine.similarity(&query, &candidate);

                if sim >= self.similarity_threshold {
                    group_indices.push(j);
                    assigned[j] = true;
                    total_sim += sim;
                    sim_count += 1;
                }
            }

            // Compute centroid
            let mut centroid = vec![0.0; self.feature_dim];
            for &idx in &group_indices {
                for (j, val) in features[idx].features.iter().enumerate().take(self.feature_dim) {
                    centroid[j] += val;
                }
            }
            for val in &mut centroid {
                *val /= group_indices.len() as f64;
            }

            // Compute average distance to centroid
            let centroid_arr = ndarray::Array1::from_vec(centroid.clone());
            let mut total_dist = 0.0;
            for &idx in &group_indices {
                let sample = ndarray::Array1::from_vec(features[idx].features.clone());
                total_dist += engine.distance(&centroid_arr, &sample);
            }
            let avg_dist = total_dist / group_indices.len() as f64;

            types.push(AcousticType {
                type_id: format!("type_{}", types.len()),
                centroid,
                count: group_indices.len(),
                sample_ids: group_indices.iter().map(|&idx| features[idx].sample_id.clone()).collect(),
                avg_distance_to_centroid: avg_dist,
            });
        }

        // Sort by count
        types.sort_by(|a, b| b.count.cmp(&a.count));

        // Build distribution
        let distribution: Vec<(String, usize)> = types.iter()
            .map(|t| (t.type_id.clone(), t.count))
            .collect();

        let avg_sim = if sim_count > 0 { total_sim / sim_count as f64 } else { 0.0 };

        (types, distribution, avg_sim)
    }

    fn build_global_types(&self, features: &[ExtractedFeatures]) -> Vec<AcousticType> {
        if features.is_empty() {
            return Vec::new();
        }

        println!("Building global type assignments from {} samples...", features.len());

        let n = features.len();

        // Create feature matrix
        let mut matrix = Array2::<f64>::zeros((n, self.feature_dim));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(self.feature_dim) {
                matrix[[i, j]] = val;
            }
        }

        // Normalize
        let mut engine = AcousticSimilarityEngine::with_metric(
            self.feature_dim,
            SimilarityMetric::Cosine,
        );
        engine.fit_normalization(&matrix);

        // Use incremental clustering with centroids
        let mut types: Vec<AcousticType> = Vec::new();
        let mut assignments = vec![0usize; n];

        println!("  Assigning samples to types (threshold: {:.2})...", self.similarity_threshold);

        for i in 0..n {
            let sample = matrix.row(i).to_owned();

            // Find best matching type
            let mut best_type = None;
            let mut best_sim = 0.0;

            for (type_idx, t) in types.iter().enumerate() {
                let centroid = ndarray::Array1::from_vec(t.centroid.clone());
                let sim = engine.similarity(&sample, &centroid);

                if sim >= self.similarity_threshold && sim > best_sim {
                    best_sim = sim;
                    best_type = Some(type_idx);
                }
            }

            if let Some(type_idx) = best_type {
                // Add to existing type
                assignments[i] = type_idx;
                types[type_idx].count += 1;
                types[type_idx].sample_ids.push(features[i].sample_id.clone());

                // Update centroid (moving average)
                let n_in_type = types[type_idx].count;
                for (j, val) in features[i].features.iter().enumerate().take(self.feature_dim) {
                    types[type_idx].centroid[j] += (val - types[type_idx].centroid[j]) / n_in_type as f64;
                }
            } else {
                // Create new type
                let type_id = format!("global_type_{}", types.len());
                assignments[i] = types.len();

                types.push(AcousticType {
                    type_id,
                    centroid: features[i].features.clone(),
                    count: 1,
                    sample_ids: vec![features[i].sample_id.clone()],
                    avg_distance_to_centroid: 0.0,
                });
            }

            if (i + 1) % 10000 == 0 {
                println!("    Processed {}/{} samples, {} types so far",
                    i + 1, n, types.len());
            }
        }

        // Sort by count
        types.sort_by(|a, b| b.count.cmp(&a.count));

        // Compute avg distances
        for t in &mut types {
            if t.count > 1 {
                let centroid = ndarray::Array1::from_vec(t.centroid.clone());
                let mut total_dist = 0.0;

                for sample_id in &t.sample_ids {
                    if let Some(idx) = features.iter().position(|f| &f.sample_id == sample_id) {
                        let sample = ndarray::Array1::from_vec(features[idx].features.clone());
                        total_dist += engine.distance(&centroid, &sample);
                    }
                }

                t.avg_distance_to_centroid = total_dist / t.count as f64;
            }
        }

        println!("  Discovered {} global types", types.len());
        println!();

        types
    }

    fn load_raw_audio(&self, path: &Path, expected_samples: usize) -> Result<Vec<f32>> {
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut buffer = Vec::with_capacity(expected_samples * 4);

        file.read_to_end(&mut buffer)?;

        let n_samples = buffer.len() / 4;
        let mut audio = Vec::with_capacity(n_samples);

        for chunk in buffer.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into()?;
            let sample = f32::from_le_bytes(bytes);
            audio.push(sample);
        }

        Ok(audio)
    }
}

// ============================================================================
// Global Assessment Functions
// ============================================================================

fn compute_global_statistics(
    features: &[ExtractedFeatures],
    types: &[AcousticType],
) -> (f64, f64, f64, f64) {
    if features.is_empty() || types.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Type entropy
    let total: usize = types.iter().map(|t| t.count).sum();
    let entropy = if total > 0 {
        types.iter()
            .map(|t| {
                let p = t.count as f64 / total as f64;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    } else {
        0.0
    };

    // Similarity statistics
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    let matrix = {
        let mut m = Array2::<f64>::zeros((features.len(), FEATURE_DIM));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };
    engine.fit_normalization(&matrix);

    // Intra-type similarity
    let mut intra_sim = 0.0;
    let mut intra_count = 0;

    for t in types.iter().take(50) {  // Sample top 50 types
        if t.count < 2 { continue; }

        let centroid = ndarray::Array1::from_vec(t.centroid.clone());
        let mut type_sim = 0.0;

        for sample_id in t.sample_ids.iter().take(10) {  // Sample 10 per type
            if let Some(f) = features.iter().find(|f| &f.sample_id == sample_id) {
                let sample = ndarray::Array1::from_vec(f.features.clone());
                type_sim += engine.similarity(&centroid, &sample);
            }
        }

        intra_sim += type_sim / t.sample_ids.len().min(10) as f64;
        intra_count += 1;
    }

    let avg_intra = if intra_count > 0 { intra_sim / intra_count as f64 } else { 0.0 };

    // Inter-type distance
    let mut inter_dist = 0.0;
    let mut inter_count = 0;

    for i in 0..types.len().min(50) {
        for j in (i+1)..types.len().min(50) {
            let a = ndarray::Array1::from_vec(types[i].centroid.clone());
            let b = ndarray::Array1::from_vec(types[j].centroid.clone());
            inter_dist += engine.distance(&a, &b);
            inter_count += 1;
        }
    }

    let avg_inter = if inter_count > 0 { inter_dist / inter_count as f64 } else { 0.0 };

    // Separation ratio
    let separation = if avg_intra > 0.0 && avg_intra < 1.0 {
        avg_inter / (1.0 - avg_intra)
    } else {
        f64::INFINITY
    };

    (entropy, avg_intra, avg_inter, separation)
}

fn evaluate_knn_classification(
    features: &[ExtractedFeatures],
) -> (f64, usize) {
    // Get primary label
    let labels: Vec<String> = features.iter()
        .map(|f| f.labels.get("source_dataset").cloned().unwrap_or_else(|| "unknown".to_string()))
        .collect();

    // Create feature matrix
    let n = features.len();
    let matrix = {
        let mut m = Array2::<f64>::zeros((n, FEATURE_DIM));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };

    // Sample for evaluation if too large
    let eval_size = n.min(10000);
    let step = n / eval_size;

    let eval_indices: Vec<usize> = (0..n).step_by(step.max(1)).take(eval_size).collect();

    let eval_features = {
        let mut m = Array2::<f64>::zeros((eval_indices.len(), FEATURE_DIM));
        for (i, &idx) in eval_indices.iter().enumerate() {
            for j in 0..FEATURE_DIM {
                m[[i, j]] = matrix[[idx, j]];
            }
        }
        m
    };

    let eval_labels: Vec<String> = eval_indices.iter().map(|&idx| labels[idx].clone()).collect();

    // Build simple k-NN using acoustic similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&eval_features);

    // Cross-validation
    let n_folds = 5;
    let fold_size = eval_size / n_folds;

    let mut total_correct = 0;
    let mut total_tested = 0;

    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 { eval_size } else { (fold + 1) * fold_size };

        for i in test_start..test_end {
            let query = eval_features.row(i).to_owned();
            let true_label = &eval_labels[i];

            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = (0..eval_size)
                .filter(|&j| j != i)
                .map(|j| {
                    let candidate = eval_features.row(j).to_owned();
                    (j, engine.distance(&query, &candidate))
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Vote
            let mut votes: HashMap<String, usize> = HashMap::new();
            for (idx, _) in distances.iter().take(K_NEIGHBORS) {
                let label = &eval_labels[*idx];
                *votes.entry(label.clone()).or_insert(0) += 1;
            }

            let predicted = votes.into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(label, _)| label)
                .unwrap_or_else(|| "unknown".to_string());

            if &predicted == true_label {
                total_correct += 1;
            }
            total_tested += 1;
        }
    }

    let accuracy = if total_tested > 0 {
        total_correct as f64 / total_tested as f64
    } else {
        0.0
    };

    (accuracy, K_NEIGHBORS)
}

fn analyze_labels(features: &[ExtractedFeatures]) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let mut source_datasets = HashMap::new();
    let mut task_types = HashMap::new();

    for f in features {
        if let Some(source) = f.labels.get("source_dataset") {
            *source_datasets.entry(source.clone()).or_insert(0) += 1;
        }
        if let Some(task) = f.labels.get("task") {
            *task_types.entry(task.clone()).or_insert(0) += 1;
        }
    }

    // Sort by count
    let mut source_vec: Vec<_> = source_datasets.into_iter().collect();
    source_vec.sort_by(|a, b| b.1.cmp(&a.1));
    source_datasets = source_vec.into_iter().collect();

    let mut task_vec: Vec<_> = task_types.into_iter().collect();
    task_vec.sort_by(|a, b| b.1.cmp(&a.1));
    task_types = task_vec.into_iter().collect();

    (source_datasets, task_types)
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   BEANS-Zero Acoustic Similarity Assessment (Chunked Processing)          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let manifest_path = PathBuf::from("beans_zero_cache/beans_audio_manifest.json");

    println!("Loading manifest: {}", manifest_path.display());
    let manifest: Manifest = {
        let file = File::open(&manifest_path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    };

    let base_path = manifest_path.parent().unwrap_or(Path::new("."));

    println!();
    println!("Configuration:");
    println!("  ├─ Dataset: {}", manifest.dataset);
    println!("  ├─ Split: {}", manifest.split);
    println!("  ├─ Total Samples: {}", manifest.samples.len());
    println!("  ├─ Chunk Size: {}", CHUNK_SIZE);
    println!("  ├─ Feature Dimension: {}D", FEATURE_DIM);
    println!("  ├─ Similarity Threshold: {:.2}", SIMILARITY_THRESHOLD);
    println!("  └─ k-NN Neighbors: {}", K_NEIGHBORS);
    println!();

    let total_start = Instant::now();

    // === Phase 1: Chunked Feature Extraction with Type Discovery ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 1: Chunked Feature Extraction with Acoustic Similarity");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let processor = ChunkedSimilarityProcessor::new(
        CHUNK_SIZE,
        manifest.resample_rate,
        FEATURE_DIM,
        SIMILARITY_THRESHOLD,
    );

    let (features, global_types, chunk_results) = processor.process_all_chunks(&manifest, base_path);

    // === Phase 2: Global Statistics ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 2: Global Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (type_entropy, avg_intra, avg_inter, separation) =
        compute_global_statistics(&features, &global_types);

    println!("Type Discovery:");
    println!("  ├─ Global Types: {}", global_types.len());
    println!("  ├─ Type Entropy: {:.3} bits", type_entropy);
    println!();
    println!("Similarity Statistics:");
    println!("  ├─ Avg Intra-Type Similarity: {:.4}", avg_intra);
    println!("  ├─ Avg Inter-Type Distance: {:.4}", avg_inter);
    println!("  └─ Separation Ratio: {:.2}x", separation);
    println!();

    // Top types
    println!("Top 10 Types by Occurrence:");
    for (i, t) in global_types.iter().take(10).enumerate() {
        println!("  {:2}. {} - {} samples, avg dist: {:.4}",
            i + 1, t.type_id, t.count, t.avg_distance_to_centroid);
    }
    println!();

    // === Phase 3: k-NN Classification ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 3: k-NN Classification (Using Acoustic Similarity Engine)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (knn_accuracy, knn_k) = evaluate_knn_classification(&features);

    println!("k-NN Results ({}-NN):", knn_k);
    println!("  └─ Accuracy: {:.2}%", knn_accuracy * 100.0);
    println!();

    // === Phase 4: Label Analysis ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 4: Label Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (source_datasets, task_types) = analyze_labels(&features);

    println!("Source Datasets ({}):", source_datasets.len());
    for (source, count) in source_datasets.iter().take(10) {
        let pct = *count as f64 / features.len() as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", source, count, pct);
    }
    if source_datasets.len() > 10 {
        println!("  └─ ... and {} more", source_datasets.len() - 10);
    }
    println!();

    println!("Task Types ({}):", task_types.len());
    for (task, count) in task_types.iter() {
        let pct = *count as f64 / features.len() as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", task, count, pct);
    }
    println!();

    // === Final Summary ===
    let total_time = total_start.elapsed().as_secs_f64();
    let throughput = features.len() as f64 / total_time;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FINAL SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let assessment = GlobalAssessment {
        dataset: manifest.dataset.clone(),
        total_samples: features.len(),
        total_chunks: chunk_results.len(),
        feature_dim: FEATURE_DIM,
        total_time_sec: total_time,
        throughput_samples_per_sec: throughput,
        global_types: global_types.len(),
        type_entropy,
        knn_accuracy,
        knn_best_k: knn_k,
        avg_intra_type_similarity: avg_intra,
        avg_inter_type_distance: avg_inter,
        separation_ratio: separation,
        source_datasets,
        task_types,
    };

    println!("Dataset: {}", assessment.dataset);
    println!("Samples processed: {}", assessment.total_samples);
    println!("Total chunks: {}", assessment.total_chunks);
    println!("Feature dimensionality: {}D", assessment.feature_dim);
    println!();

    println!("Performance:");
    println!("  ├─ Total time: {:.2}s ({:.1} min)", assessment.total_time_sec, assessment.total_time_sec / 60.0);
    println!("  ├─ Throughput: {:.1} samples/sec", assessment.throughput_samples_per_sec);
    println!("  └─ Chunks processed: {}", assessment.total_chunks);
    println!();

    println!("Type Discovery (Acoustic Similarity):");
    println!("  ├─ Global types: {}", assessment.global_types);
    println!("  ├─ Type entropy: {:.3} bits", assessment.type_entropy);
    println!("  ├─ Intra-type similarity: {:.4}", assessment.avg_intra_type_similarity);
    println!("  ├─ Inter-type distance: {:.4}", assessment.avg_inter_type_distance);
    println!("  └─ Separation ratio: {:.2}x", assessment.separation_ratio);
    println!();

    println!("Classification (k-NN with Acoustic Similarity):");
    println!("  ├─ Best k: {}-NN", assessment.knn_best_k);
    println!("  └─ Accuracy: {:.2}%", assessment.knn_accuracy * 100.0);
    println!();

    // Determine competence level
    let competence = if assessment.knn_accuracy > 0.8 && assessment.separation_ratio > 2.0 {
        "EXCELLENT"
    } else if assessment.knn_accuracy > 0.7 && assessment.separation_ratio > 1.5 {
        "GOOD"
    } else if assessment.knn_accuracy > 0.6 {
        "MODERATE"
    } else {
        "NEEDS IMPROVEMENT"
    };

    println!("Competence Level: {}", competence);
    println!();

    // Save results
    let output_dir = PathBuf::from("beans_analysis");
    std::fs::create_dir_all(&output_dir)?;

    let results_path = output_dir.join("beans_acoustic_similarity_results.json");
    let file = File::create(&results_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &assessment)?;

    println!("Output files:");
    println!("  └─ Results: {}", results_path.display());

    Ok(())
}
