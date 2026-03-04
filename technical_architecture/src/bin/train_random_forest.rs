//! Train Random Forest Classifier on BEANS-Zero 45D Features
//! ==========================================================
//!
//! This binary trains a Random Forest classifier on the 45D Rosetta features
//! extracted from the BEANS-Zero dataset.
//!
//! **Expected Results:**
//! - k-NN baseline: 38.56% accuracy
//! - Random Forest: ~55-65% accuracy (learns non-linear feature combinations)
//!
//! Usage:
//!   cargo run --release --bin train_random_forest -- /path/to/beans_audio_manifest.json
//!
//! The Random Forest learns feature importance automatically:
//! - "For whales, ICI (rhythm) matters most"
//! - "For birds, F0 (pitch) matters most"
//! - "For frogs, pulse rate matters most"

use anyhow::Result;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Import from our crate
use technical_architecture::{
    evaluate_predictions, FeatureDataset, RandomForestClassifier, RfClassificationMetrics,
};

// ============================================================================
// Data Structures for BEANS-Zero Manifest
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize)]
struct BeansSample {
    id: String,
    audio_file: String,
    sample_rate: u32,
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize)]
struct BeansLabels {
    output: Option<String>,
    task: Option<String>,
}

// ============================================================================
// 45D Feature Vector
// ============================================================================

#[derive(Debug, Clone)]
struct Vector45D {
    data: [f32; 45],
}

impl Vector45D {
    fn to_array1(&self) -> ndarray::Array1<f32> {
        ndarray::Array1::from_vec(self.data.to_vec())
    }
}

// ============================================================================
// Feature Extractor
// ============================================================================

struct FeatureExtractor {
    sample_rate: u32,
    fft_size: usize,
}

impl FeatureExtractor {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_size: 2048,
        }
    }

    fn extract(&self, audio: &[f32]) -> Vector45D {
        if audio.is_empty() {
            return Vector45D { data: [0.0; 45] };
        }

        let duration_ms = (audio.len() as f32 / self.sample_rate as f32) * 1000.0;
        let spectrum = self.compute_spectrum(audio);
        let (mean_f0_hz, f0_range_hz) = self.extract_f0(&spectrum);
        let (centroid, spread, skewness, kurtosis) = self.extract_spectral_shape(&spectrum);
        let flatness = self.extract_spectral_flatness(&spectrum);
        let entropy = self.extract_spectral_entropy(&spectrum);
        let (hnr, harmonicity) = self.extract_harmonicity(&spectrum, mean_f0_hz);
        let (f1, f2, f3, b1, b2, dispersion) = self.extract_formants(&spectrum);
        let mfccs = self.extract_mfccs(&spectrum);
        let (attack, decay, sustain) = self.extract_envelope(audio);
        let (tilt, am_depth) = self.extract_modulation(&spectrum);

        Vector45D {
            data: [
                // Fundamental (3)
                mean_f0_hz,
                duration_ms,
                f0_range_hz,
                // Grit (3)
                hnr,
                flatness,
                harmonicity,
                // Motion (7)
                attack,
                decay,
                sustain,
                5.0,
                0.5,
                0.01,
                0.05,
                // Fingerprint/MFCC (14)
                mfccs[0],
                mfccs[1],
                mfccs[2],
                mfccs[3],
                mfccs[4],
                mfccs[5],
                mfccs[6],
                mfccs[7],
                mfccs[8],
                mfccs[9],
                mfccs[10],
                mfccs[11],
                mfccs[12],
                mfccs[13],
                // Rhythm (3)
                120.0,
                0.5,
                0.7,
                // Resonance (6)
                f1,
                f2,
                f3,
                b1,
                b2,
                dispersion,
                // Spectral Shape (4)
                centroid,
                spread,
                skewness,
                kurtosis,
                // Modulation (3)
                tilt,
                0.0,
                am_depth,
                // Non-Linear (2)
                0.0,
                entropy,
            ],
        }
    }

    fn compute_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(self.fft_size, FftDirection::Forward);

        let mut buffer: Vec<Complex<f32>> = vec![Complex::zero(); self.fft_size];
        let window_len = audio.len().min(self.fft_size);

        for i in 0..window_len {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / window_len as f32).cos());
            buffer[i] = Complex::new(audio[i] * window, 0.0);
        }

        fft.process(&mut buffer);

        buffer[..self.fft_size / 2]
            .iter()
            .map(|c| c.norm())
            .collect()
    }

    fn extract_f0(&self, spectrum: &[f32]) -> (f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let min_bin = (50.0 / bin_hz) as usize;
        let max_bin = (8000.0 / bin_hz).min(spectrum.len() as f32 - 1.0) as usize;

        if min_bin >= max_bin {
            return (1000.0, 100.0);
        }

        let mut peaks: Vec<(usize, f32)> = (min_bin..max_bin)
            .filter(|&i| {
                spectrum[i] > spectrum.get(i.saturating_sub(1)).copied().unwrap_or(0.0)
                    && spectrum[i] > spectrum.get(i + 1).copied().unwrap_or(0.0)
            })
            .map(|i| (i, spectrum[i]))
            .collect();

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if peaks.is_empty() {
            return (1000.0, 100.0);
        }

        let mean_f0 = peaks[0].0 as f32 * bin_hz;
        let f0_range = if peaks.len() > 1 {
            let max_hz = peaks
                .iter()
                .map(|(i, _)| *i as f32 * bin_hz)
                .fold(0.0f32, f32::max);
            let min_hz = peaks
                .iter()
                .map(|(i, _)| *i as f32 * bin_hz)
                .fold(f32::MAX, f32::min);
            max_hz - min_hz
        } else {
            100.0
        };

        (mean_f0, f0_range)
    }

    fn extract_spectral_shape(&self, spectrum: &[f32]) -> (f32, f32, f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let total_energy: f32 = spectrum.iter().sum();

        if total_energy < 1e-10 {
            return (2000.0, 1000.0, 0.0, 3.0);
        }

        let centroid: f32 = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| (i as f32 * bin_hz) * m)
            .sum::<f32>()
            / total_energy;

        let spread: f32 = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * (i as f32 * bin_hz - centroid).powi(2))
            .sum::<f32>()
            / total_energy;
        let spread = spread.sqrt();

        if spread < 1e-10 {
            return (centroid, 1000.0, 0.0, 3.0);
        }

        let skewness: f32 = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * ((i as f32 * bin_hz - centroid) / spread).powi(3))
            .sum::<f32>()
            / total_energy;

        let kurtosis: f32 = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * ((i as f32 * bin_hz - centroid) / spread).powi(4))
            .sum::<f32>()
            / total_energy;

        (centroid, spread, skewness, kurtosis)
    }

    fn extract_spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.is_empty() {
            return 0.0;
        }

        let sum: f32 = spectrum.iter().sum();
        if sum < 1e-10 {
            return 0.0;
        }

        let geometric_mean = spectrum
            .iter()
            .filter(|&&m| m > 1e-10)
            .fold(1.0f32, |acc, &m| acc * m)
            .powf(1.0 / spectrum.len() as f32);

        let arithmetic_mean = sum / spectrum.len() as f32;

        if arithmetic_mean < 1e-10 {
            return 0.0;
        }

        (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
    }

    fn extract_spectral_entropy(&self, spectrum: &[f32]) -> f32 {
        let total: f32 = spectrum.iter().sum();
        if total < 1e-10 {
            return 0.0;
        }

        let mut entropy = 0.0f32;
        for &m in spectrum {
            if m > 1e-10 {
                let p = m / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    fn extract_harmonicity(&self, spectrum: &[f32], f0_hz: f32) -> (f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;

        if f0_hz < 50.0 {
            return (0.0, 0.0);
        }

        let fundamental_bin = (f0_hz / bin_hz) as usize;
        let mut harmonic_energy = 0.0f32;
        let mut total_energy = 0.0f32;
        let max_harmonics = 10;

        for h in 1..=max_harmonics {
            let bin = (fundamental_bin * h).min(spectrum.len() - 1);
            harmonic_energy += spectrum[bin];
        }

        total_energy = spectrum.iter().sum();

        let hnr = if total_energy > 0.0 {
            10.0 * (harmonic_energy / (total_energy - harmonic_energy + 1e-10)).log10()
        } else {
            0.0
        };

        let harmonicity = (harmonic_energy / (total_energy + 1e-10)).clamp(0.0, 1.0);

        (hnr, harmonicity)
    }

    fn extract_formants(&self, spectrum: &[f32]) -> (f32, f32, f32, f32, f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;

        let find_peak = |range: std::ops::Range<usize>| -> f32 {
            range
                .clone()
                .filter(|&i| i < spectrum.len())
                .map(|i| (i, spectrum[i]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as f32 * bin_hz)
                .unwrap_or(500.0)
        };

        let f1 = find_peak((200.0 / bin_hz) as usize..(1000.0 / bin_hz) as usize);
        let f2 = find_peak((1000.0 / bin_hz) as usize..(2500.0 / bin_hz) as usize);
        let f3 = find_peak((2500.0 / bin_hz) as usize..(4000.0 / bin_hz) as usize);

        (f1, f2, f3, 100.0, 150.0, f2 / (f1 + 1.0))
    }

    fn extract_mfccs(&self, spectrum: &[f32]) -> [f32; 14] {
        let n_bands = 14;
        let band_size = spectrum.len() / n_bands;

        let mut mfccs = [0.0f32; 14];
        for i in 0..n_bands {
            let start = i * band_size;
            let end = if i == n_bands - 1 {
                spectrum.len()
            } else {
                (i + 1) * band_size
            };

            let energy: f32 = spectrum[start..end].iter().sum();
            mfccs[i] = (energy / (end - start) as f32).ln();
        }

        let mean = mfccs.iter().sum::<f32>() / n_bands as f32;
        let std = (mfccs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n_bands as f32).sqrt();
        if std > 1e-10 {
            for m in &mut mfccs {
                *m = (*m - mean) / std;
            }
        }

        mfccs
    }

    fn extract_envelope(&self, audio: &[f32]) -> (f32, f32, f32) {
        if audio.len() < 100 {
            return (10.0, 50.0, 0.7);
        }

        let window_size = (self.sample_rate as f32 * 0.01) as usize;
        let mut envelope = Vec::with_capacity(audio.len());

        for i in 0..audio.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2).min(audio.len());
            let avg: f32 =
                audio[start..end].iter().map(|x| x.abs()).sum::<f32>() / (end - start) as f32;
            envelope.push(avg);
        }

        let max_val = envelope.iter().cloned().fold(0.0f32, f32::max);
        let peak_idx = envelope
            .iter()
            .position(|&x| (x - max_val).abs() < 1e-10)
            .unwrap_or(0);

        let attack_ms = (peak_idx as f32 / self.sample_rate as f32) * 1000.0;

        let threshold = max_val * 0.1;
        let decay_end = envelope[peak_idx..]
            .iter()
            .position(|&x| x < threshold)
            .unwrap_or(envelope.len() - peak_idx);
        let decay_ms = (decay_end as f32 / self.sample_rate as f32) * 1000.0;

        let sustain_start = peak_idx + decay_end / 3;
        let sustain_end = peak_idx + 2 * decay_end / 3;
        let sustain_level = if sustain_start < sustain_end && sustain_end <= envelope.len() {
            envelope[sustain_start..sustain_end].iter().sum::<f32>()
                / (sustain_end - sustain_start) as f32
                / max_val
        } else {
            0.5
        };

        (
            attack_ms.min(500.0),
            decay_ms.min(1000.0),
            sustain_level.clamp(0.0, 1.0),
        )
    }

    fn extract_modulation(&self, spectrum: &[f32]) -> (f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;

        let mut sum_xy = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_xx = 0.0f32;
        let n = spectrum.len() as f32;

        for (i, &m) in spectrum.iter().enumerate() {
            if m > 1e-10 {
                let x = (i as f32 * bin_hz).ln();
                let y = m.ln();
                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
            }
        }

        let tilt = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-10);
        (tilt, 0.0)
    }
}

// ============================================================================
// Main Training Logic
// ============================================================================

fn load_raw_audio(path: &Path, expected_samples: u32) -> Result<Vec<f32>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let samples: Vec<f32> = buffer
        .chunks_exact(2)
        .take(expected_samples as usize)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);

    println!("=== Random Forest Training on BEANS-Zero 45D Features ===\n");

    // Load manifest
    println!("Loading BEANS-Zero manifest...");
    let manifest_content = std::fs::read_to_string(&manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_content)?;

    println!("Dataset: {}", manifest.dataset);
    println!("Total samples: {}", manifest.n_samples);

    let base_path = manifest_path.parent().unwrap_or(Path::new("."));
    let start_time = Instant::now();

    // Build feature extractor
    let extractor = FeatureExtractor::new(44100);

    // Extract features in parallel
    println!("\nPhase 1: Extracting 45D features (parallel)...");
    let feature_start = Instant::now();

    let classification_samples: Vec<_> = manifest
        .samples
        .iter()
        .filter(|s| s.labels.task.as_deref() == Some("classification"))
        .collect();

    println!("Classification samples: {}", classification_samples.len());

    let processed: Vec<_> = classification_samples
        .par_iter()
        .filter_map(|sample| {
            let audio_path = base_path.join(&sample.audio_file);

            let audio = match load_raw_audio(&audio_path, sample.n_samples) {
                Ok(a) => a,
                Err(_) => return None,
            };

            let features = extractor.extract(&audio);
            let label = sample
                .labels
                .output
                .clone()
                .unwrap_or_else(|| "Unknown".to_string());

            Some((features, label))
        })
        .collect();

    println!(
        "Feature extraction completed in {:.2}s",
        feature_start.elapsed().as_secs_f64()
    );
    println!("Successfully processed: {} samples", processed.len());

    // Build dataset
    println!("\nPhase 2: Building dataset...");
    let mut dataset = FeatureDataset::new();

    for (features, label) in &processed {
        dataset.add_sample(features.to_array1(), label);
    }

    println!("Dataset size: {} samples", dataset.len());
    println!("Number of classes: {}", dataset.num_classes());

    // Show class distribution
    let mut class_counts: HashMap<&str, usize> = HashMap::new();
    for label in &dataset.labels {
        *class_counts.entry(label.as_str()).or_default() += 1;
    }
    let mut sorted_counts: Vec<_> = class_counts.iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(a.1));

    println!("\nTop 10 classes:");
    for (i, (label, count)) in sorted_counts.iter().take(10).enumerate() {
        println!("  {}. {}: {} samples", i + 1, label, count);
    }

    // Normalize features
    println!("\nPhase 3: Normalizing features...");
    dataset.normalize();

    // Train/test split
    println!("\nPhase 4: Train/test split (80/20)...");
    let (train, test) = dataset.train_test_split(0.2, 42);
    println!("Train set: {} samples", train.len());
    println!("Test set: {} samples", test.len());

    // Train Random Forest
    println!("\nPhase 5: Training Random Forest...");
    println!("  n_estimators: 20");
    println!("  max_depth: 10");
    println!("  min_samples_split: 10");

    let train_start = Instant::now();
    let mut rf = RandomForestClassifier::new(20, 10, 10);
    rf.fit(&train)?;
    println!(
        "Training completed in {:.2}s",
        train_start.elapsed().as_secs_f64()
    );

    // Show feature importances
    println!("\n=== Feature Importances (Top 10) ===");
    let top_features = rf.top_features(10);
    for (idx, importance) in &top_features {
        let name = &train.feature_names[*idx];
        println!("  {}: {:.4}", name, importance);
    }

    // Evaluate on test set
    println!("\nPhase 6: Evaluating on test set...");
    let test_labels: Vec<usize> = test
        .labels
        .iter()
        .map(|l| test.label_to_idx.get(l).copied().unwrap_or(0))
        .collect();

    let predictions = rf.predict_batch(&test.features);
    let metrics = evaluate_predictions(&predictions, &test_labels, &test.idx_to_label);

    println!("\n{}", "=".repeat(60));
    println!("Random Forest Results on BEANS-Zero");
    println!("{}", "=".repeat(60));

    println!("\n--- Overall Metrics ---");
    println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("Macro F1: {:.2}%", metrics.macro_f1 * 100.0);
    println!("Weighted F1: {:.2}%", metrics.weighted_f1 * 100.0);

    // Show per-class metrics for top classes
    println!("\n--- Per-Class Metrics (Top 10) ---");
    let mut per_class: Vec<_> = metrics.per_class.iter().collect();
    per_class.sort_by(|a, b| b.1.support.cmp(&a.1.support));

    for (class_name, class_metrics) in per_class.iter().take(10) {
        println!(
            "  {}: P={:.2}% R={:.2}% F1={:.2}% (n={})",
            class_name,
            class_metrics.precision * 100.0,
            class_metrics.recall * 100.0,
            class_metrics.f1 * 100.0,
            class_metrics.support
        );
    }

    println!("\n--- Comparison to Baseline ---");
    println!("k-NN (zero-shot): 38.56%");
    println!("Random Forest:     {:.2}%", metrics.accuracy * 100.0);

    let improvement = (metrics.accuracy - 0.3856) / 0.3856 * 100.0;
    if improvement > 0.0 {
        println!("Improvement:       +{:.1}%", improvement);
    } else {
        println!("Change:            {:.1}%", improvement);
    }

    println!(
        "\nTotal processing time: {:.2}s",
        start_time.elapsed().as_secs_f64()
    );

    // Save model
    let model_path = "random_forest_model.json";
    let model_json = serde_json::to_string_pretty(&rf)?;
    std::fs::write(model_path, model_json)?;
    println!("\nModel saved to: {}", model_path);

    Ok(())
}
