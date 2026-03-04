//! Train Balanced Random Forest and Hierarchical Classifier
//! =========================================================
//!
//! This binary implements the recommended improvements to address class imbalance:
//!
//! **Method A: Balanced Random Forest**
//! Uses class weighting to oversample minority classes during bootstrap sampling.
//!
//! **Method B: Hierarchical Classification**
//! Level 1: Predict Taxonomic Group (Cetacean, Bird, Insect, etc.)
//! Level 2: Predict Species within Group
//!
//! **Expected Improvement:**
//! - Standard RF: 47.85%
//! - Balanced RF: ~50-55% (better minority class recall)
//! - Hierarchical: ~55-60% (leverages taxonomic structure)
//!
//! Usage:
//!   cargo run --release --bin train_balanced_rf -- /path/to/beans_audio_manifest.json

use anyhow::Result;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use technical_architecture::{
    evaluate_predictions, ClassWeightMode, FeatureDataset, HierarchicalClassifier,
    RandomForestClassifier, RfClassificationMetrics, TaxonomicGroup,
};

// ============================================================================
// Data Structures (same as train_random_forest.rs)
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize)]
struct BeansSample {
    audio_file: String,
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize)]
struct BeansLabels {
    output: Option<String>,
    task: Option<String>,
}

#[derive(Debug, Clone)]
struct Vector45D {
    data: [f32; 45],
}

// ============================================================================
// Feature Extractor (simplified)
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
                mean_f0_hz,
                duration_ms,
                f0_range_hz,
                hnr,
                flatness,
                harmonicity,
                attack,
                decay,
                sustain,
                5.0,
                0.5,
                0.01,
                0.05,
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
                120.0,
                0.5,
                0.7,
                f1,
                f2,
                f3,
                b1,
                b2,
                dispersion,
                centroid,
                spread,
                skewness,
                kurtosis,
                tilt,
                0.0,
                am_depth,
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

        let centroid = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| (i as f32 * bin_hz) * m)
            .sum::<f32>()
            / total_energy;
        let spread = (spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * (i as f32 * bin_hz - centroid).powi(2))
            .sum::<f32>()
            / total_energy)
            .sqrt();
        if spread < 1e-10 {
            return (centroid, 1000.0, 0.0, 3.0);
        }

        let skewness = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * ((i as f32 * bin_hz - centroid) / spread).powi(3))
            .sum::<f32>()
            / total_energy;
        let kurtosis = spectrum
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
        if f0_hz < 50.0 {
            return (0.0, 0.0);
        }
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let fundamental_bin = (f0_hz / bin_hz) as usize;
        let mut harmonic_energy = 0.0f32;
        for h in 1..=10 {
            let bin = (fundamental_bin * h).min(spectrum.len() - 1);
            harmonic_energy += spectrum[bin];
        }
        let total_energy: f32 = spectrum.iter().sum();
        let hnr = if total_energy > 0.0 {
            10.0 * (harmonic_energy / (total_energy - harmonic_energy + 1e-10)).log10()
        } else {
            0.0
        };
        (
            hnr,
            (harmonic_energy / (total_energy + 1e-10)).clamp(0.0, 1.0),
        )
    }

    fn extract_formants(&self, spectrum: &[f32]) -> (f32, f32, f32, f32, f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let find_peak = |range: std::ops::Range<usize>| -> f32 {
            range
                .clone()
                .filter(|&i| i < spectrum.len())
                .max_by(|a, b| {
                    spectrum[*a]
                        .partial_cmp(&spectrum[*b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|i| i as f32 * bin_hz)
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
        let envelope: Vec<f32> = (0..audio.len())
            .map(|i| {
                let start = i.saturating_sub(window_size / 2);
                let end = (i + window_size / 2).min(audio.len());
                audio[start..end].iter().map(|x| x.abs()).sum::<f32>() / (end - start) as f32
            })
            .collect();
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
        (attack_ms.min(500.0), decay_ms.min(1000.0), 0.5)
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
        (
            (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-10),
            0.0,
        )
    }
}

fn load_raw_audio(path: &Path, expected_samples: u32) -> Result<Vec<f32>> {
    use std::fs::File;
    use std::io::Read;
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer
        .chunks_exact(2)
        .take(expected_samples as usize)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    println!("=== Balanced Random Forest & Hierarchical Classification ===\n");

    // Load manifest
    println!("Loading BEANS-Zero manifest...");
    let manifest: BeansManifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    println!("Dataset: {}", manifest.dataset);
    let base_path = manifest_path.parent().unwrap_or(Path::new("."));
    let start_time = Instant::now();

    // Extract features
    println!("\nPhase 1: Extracting 45D features...");
    let extractor = FeatureExtractor::new(44100);
    let classification_samples: Vec<_> = manifest
        .samples
        .iter()
        .filter(|s| s.labels.task.as_deref() == Some("classification"))
        .collect();

    let processed: Vec<_> = classification_samples
        .par_iter()
        .filter_map(|sample| {
            let audio_path = base_path.join(&sample.audio_file);
            let audio = load_raw_audio(&audio_path, sample.n_samples).ok()?;
            let features = extractor.extract(&audio);
            let label = sample
                .labels
                .output
                .clone()
                .unwrap_or_else(|| "Unknown".to_string());
            Some((features, label))
        })
        .collect();
    println!("Processed: {} samples", processed.len());

    // Build dataset
    println!("\nPhase 2: Building dataset...");
    let mut dataset = FeatureDataset::new();
    for (features, label) in &processed {
        dataset.add_sample(ndarray::Array1::from_vec(features.data.to_vec()), label);
    }

    // Analyze class distribution
    let mut class_counts: HashMap<&str, usize> = HashMap::new();
    for label in &dataset.labels {
        *class_counts.entry(label.as_str()).or_default() += 1;
    }
    println!("Classes: {}", class_counts.len());

    // Analyze taxonomic groups
    let mut group_counts: HashMap<TaxonomicGroup, usize> = HashMap::new();
    for label in &dataset.labels {
        let group = TaxonomicGroup::from_species_name(label);
        *group_counts.entry(group).or_default() += 1;
    }
    println!("\nTaxonomic Groups:");
    for (group, count) in &group_counts {
        println!("  {:?}: {} samples", group, count);
    }

    // Normalize
    println!("\nPhase 3: Normalizing features...");
    dataset.normalize();

    // Split
    println!("\nPhase 4: Train/test split (80/20)...");
    let (train, test) = dataset.train_test_split(0.2, 42);
    println!("Train: {} | Test: {}", train.len(), test.len());

    // Method A: Standard Random Forest (baseline)
    println!("\n{}", "=".repeat(60));
    println!("Method A: Standard Random Forest (Baseline)");
    println!("{}", "=".repeat(60));
    let mut rf_standard = RandomForestClassifier::new(20, 10, 10);
    rf_standard.fit(&train)?;
    let predictions = rf_standard.predict_batch(&test.features);
    let test_labels: Vec<usize> = test
        .labels
        .iter()
        .map(|l| test.label_to_idx.get(l).copied().unwrap_or(0))
        .collect();
    let metrics_standard = evaluate_predictions(&predictions, &test_labels, &test.idx_to_label);
    println!("Accuracy: {:.2}%", metrics_standard.accuracy * 100.0);

    // Method B: Balanced Random Forest
    println!("\n{}", "=".repeat(60));
    println!("Method B: Balanced Random Forest (Class Weighting)");
    println!("{}", "=".repeat(60));
    let mut rf_balanced = RandomForestClassifier::new(20, 10, 10).with_balanced_weights();
    rf_balanced.fit(&train)?;
    let predictions = rf_balanced.predict_batch(&test.features);
    let metrics_balanced = evaluate_predictions(&predictions, &test_labels, &test.idx_to_label);
    println!("Accuracy: {:.2}%", metrics_balanced.accuracy * 100.0);

    // Method C: Hierarchical Classification
    println!("\n{}", "=".repeat(60));
    println!("Method C: Hierarchical Classification");
    println!("{}", "=".repeat(60));
    let mut hierarchical = HierarchicalClassifier::new(20, 10, 10);
    hierarchical.fit(&train)?;
    let metrics_hierarchical = hierarchical.evaluate(&test);
    println!("Accuracy: {:.2}%", metrics_hierarchical.accuracy * 100.0);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY: Comparison of Methods");
    println!("{}", "=".repeat(60));
    println!("k-NN baseline:        38.56%");
    println!(
        "Standard RF:          {:.2}%",
        metrics_standard.accuracy * 100.0
    );
    println!(
        "Balanced RF:          {:.2}%",
        metrics_balanced.accuracy * 100.0
    );
    println!(
        "Hierarchical:         {:.2}%",
        metrics_hierarchical.accuracy * 100.0
    );

    let best = metrics_balanced
        .accuracy
        .max(metrics_standard.accuracy)
        .max(metrics_hierarchical.accuracy);
    let improvement = (best - 0.3856) / 0.3856 * 100.0;
    println!(
        "\nBest improvement:      +{:.1}% over baseline",
        improvement
    );
    println!("\nTotal time: {:.2}s", start_time.elapsed().as_secs_f64());

    Ok(())
}
