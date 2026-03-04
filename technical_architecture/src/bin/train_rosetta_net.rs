//! Train Rosetta-Net with TCN on BEANS-Zero
//! ==========================================
//!
//! Training strategy based on Random Forest insights:
//! - duration_ms is 74% important
//! - TCN captures temporal dynamics (duration, attack, decay, rhythm)
//! - Heavy weight on 45D Rosetta regression forces learning "physics"
//!
//! Loss Function:
//!   L = α * MSE(rosetta_pred, rosetta_target) + β * CrossEntropy(logits, class)
//!
//! Where α >> β to ensure the network learns the physics first.
//!
//! Expected Results:
//! - If 45D MSE is low, classification accuracy will follow
//! - Target: Beat 47.85% Random Forest baseline
//!
//! Usage:
//!   cargo run --release --bin train_rosetta_net -- /path/to/beans_audio_manifest.json

use anyhow::Result;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use technical_architecture::{
    cross_entropy_loss, mse_loss, EncoderType, RosettaNetConfig, RosettaNetWithTCN, Spectrogram,
};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
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

// ============================================================================
// Feature Extractor (45D)
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

    fn extract(&self, audio: &[f32]) -> [f32; 45] {
        if audio.is_empty() {
            return [0.0; 45];
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

        [
            // Fundamental (3) - indices 0-2
            mean_f0_hz,
            duration_ms,
            f0_range_hz,
            // Grit (3) - indices 3-5
            hnr,
            flatness,
            harmonicity,
            // Motion (7) - indices 6-12
            attack,
            decay,
            sustain,
            5.0,
            0.5,
            0.01,
            0.05,
            // Fingerprint/MFCC (14) - indices 13-26
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
            // Rhythm (3) - indices 27-29
            120.0,
            0.5,
            0.7,
            // Resonance (6) - indices 30-35
            f1,
            f2,
            f3,
            b1,
            b2,
            dispersion,
            // Spectral Shape (4) - indices 36-39
            centroid,
            spread,
            skewness,
            kurtosis,
            // Modulation (3) - indices 40-42
            tilt,
            0.0,
            am_depth,
            // Non-Linear (2) - indices 43-44
            0.0,
            entropy,
        ]
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
        (peaks[0].0 as f32 * bin_hz, 100.0)
    }

    fn extract_spectral_shape(&self, spectrum: &[f32]) -> (f32, f32, f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let total: f32 = spectrum.iter().sum();
        if total < 1e-10 {
            return (2000.0, 1000.0, 0.0, 3.0);
        }

        let centroid = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| (i as f32 * bin_hz) * m)
            .sum::<f32>()
            / total;
        let spread = (spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * (i as f32 * bin_hz - centroid).powi(2))
            .sum::<f32>()
            / total)
            .sqrt();
        if spread < 1e-10 {
            return (centroid, 1000.0, 0.0, 3.0);
        }

        let skew = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * ((i as f32 * bin_hz - centroid) / spread).powi(3))
            .sum::<f32>()
            / total;
        let kurt = spectrum
            .iter()
            .enumerate()
            .map(|(i, &m)| m * ((i as f32 * bin_hz - centroid) / spread).powi(4))
            .sum::<f32>()
            / total;
        (centroid, spread, skew, kurt)
    }

    fn extract_spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.is_empty() {
            return 0.0;
        }
        let sum: f32 = spectrum.iter().sum();
        if sum < 1e-10 {
            return 0.0;
        }
        let gm = spectrum
            .iter()
            .filter(|&&m| m > 1e-10)
            .fold(1.0f32, |acc, &m| acc * m)
            .powf(1.0 / spectrum.len() as f32);
        let am = sum / spectrum.len() as f32;
        if am < 1e-10 {
            return 0.0;
        }
        (gm / am).clamp(0.0, 1.0)
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
        let fund_bin = (f0_hz / bin_hz) as usize;
        let mut harmonic_energy = 0.0f32;
        for h in 1..=10 {
            let bin = (fund_bin * h).min(spectrum.len() - 1);
            harmonic_energy += spectrum[bin];
        }
        let total: f32 = spectrum.iter().sum();
        let hnr = if total > 0.0 {
            10.0 * (harmonic_energy / (total - harmonic_energy + 1e-10)).log10()
        } else {
            0.0
        };
        (hnr, (harmonic_energy / (total + 1e-10)).clamp(0.0, 1.0))
    }

    fn extract_formants(&self, spectrum: &[f32]) -> (f32, f32, f32, f32, f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let find_peak = |r: std::ops::Range<usize>| -> f32 {
            r.clone()
                .filter(|&i| i < spectrum.len())
                .max_by(|a, b| {
                    spectrum[*a]
                        .partial_cmp(&spectrum[*b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|i| i as f32 * bin_hz)
                .unwrap_or(500.0)
        };
        (
            find_peak((200. / bin_hz) as usize..(1000. / bin_hz) as usize),
            find_peak((1000. / bin_hz) as usize..(2500. / bin_hz) as usize),
            find_peak((2500. / bin_hz) as usize..(4000. / bin_hz) as usize),
            100.0,
            150.0,
            1.5,
        )
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
        (10.0, 50.0, 0.5) // Simplified
    }

    fn extract_modulation(&self, spectrum: &[f32]) -> (f32, f32) {
        let bin_hz = self.sample_rate as f32 / self.fft_size as f32;
        let n = spectrum.len() as f32;
        let (mut sxy, mut sx, mut sy, mut sxx) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (i, &m) in spectrum.iter().enumerate() {
            if m > 1e-10 {
                let x = (i as f32 * bin_hz).ln();
                let y = m.ln();
                sxy += x * y;
                sx += x;
                sy += y;
                sxx += x * x;
            }
        }
        ((n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-10), 0.0)
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
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect())
}

// ============================================================================
// Training Functions
// ============================================================================

/// Compute combined loss with heavy weighting on Rosetta regression
fn compute_loss(
    rosetta_pred: &ndarray::Array1<f32>,
    rosetta_target: &ndarray::Array1<f32>,
    logits: &ndarray::Array1<f32>,
    target_class: usize,
    rosetta_weight: f32,
    class_weight: f32,
) -> (f32, f32, f32) {
    let rosetta_loss = mse_loss(rosetta_pred, rosetta_target);
    let class_loss = cross_entropy_loss(logits, target_class);
    let total = rosetta_weight * rosetta_loss + class_weight * class_loss;
    (total, rosetta_loss, class_loss)
}

/// Simple numerical gradient estimation for weight updates
/// Uses finite differences: ∂f/∂w ≈ (f(w+ε) - f(w-ε)) / (2ε)
fn estimate_gradient(
    model: &mut RosettaNetWithTCN,
    spectrogram: &ndarray::Array2<f32>,
    rosetta_target: &ndarray::Array1<f32>,
    target_class: usize,
    rosetta_weight: f32,
    class_weight: f32,
    epsilon: f32,
) -> f32 {
    // This is a placeholder - in production, you'd use autograd (tch-rs, burn, etc.)
    // For now, we just return the current loss for monitoring
    let (latent, rosetta_pred, logits) = model.forward(spectrogram);
    let (total, _, _) = compute_loss(
        &rosetta_pred,
        rosetta_target,
        &logits,
        target_class,
        rosetta_weight,
        class_weight,
    );
    total
}

// ============================================================================
// Main Training Loop
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Rosetta-Net with TCN Training on BEANS-Zero            ║");
    println!("║     \"Learning the Physics of Bioacoustics\"                 ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Load manifest
    println!("Loading BEANS-Zero manifest...");
    let manifest: BeansManifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    println!("Dataset: {}", manifest.dataset);
    let base_path = manifest_path.parent().unwrap_or(Path::new("."));
    let start_time = Instant::now();

    // Extract features and create spectrograms
    println!("\nPhase 1: Extracting features and spectrograms...");
    let feature_extractor = FeatureExtractor::new(44100);

    let classification_samples: Vec<_> = manifest
        .samples
        .iter()
        .filter(|s| s.labels.task.as_deref() == Some("classification"))
        .collect();
    println!("Classification samples: {}", classification_samples.len());

    // Process smaller subset for training (memory constraint)
    let max_samples = 1000usize.min(classification_samples.len());
    println!("Processing {} samples for training...", max_samples);

    // Process sequentially to avoid memory issues
    let mut processed = Vec::new();
    for sample in classification_samples.iter().take(max_samples) {
        let audio_path = base_path.join(&sample.audio_file);
        let audio = match load_raw_audio(&audio_path, sample.n_samples) {
            Ok(a) => a,
            Err(_) => continue,
        };

        // Skip very short or very long audio
        if audio.len() < 1000 || audio.len() > 44100 * 10 {
            continue;
        }

        // Create spectrogram
        let mut spec = Spectrogram::from_audio(&audio, 44100, 512, 1024);
        spec.normalize();
        let spec_resized = spec.resize((64, 64)); // Smaller size for memory

        // Extract 45D features as target
        let features = feature_extractor.extract(&audio);

        let label = sample
            .labels
            .output
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());

        processed.push((spec_resized, features, label));

        if processed.len() % 100 == 0 {
            println!("  Processed {} samples...", processed.len());
        }
    }

    println!("Successfully processed: {} samples", processed.len());

    // Build label mappings
    let mut label_to_idx: HashMap<String, usize> = HashMap::new();
    let mut idx_to_label: HashMap<usize, String> = HashMap::new();
    for (_, _, label) in &processed {
        if !label_to_idx.contains_key(label) {
            let idx = label_to_idx.len();
            label_to_idx.insert(label.clone(), idx);
            idx_to_label.insert(idx, label.clone());
        }
    }
    let num_classes = label_to_idx.len();
    println!("Number of classes: {}", num_classes);

    // Split into train/val
    println!("\nPhase 2: Train/validation split (80/20)...");
    let n = processed.len();
    let n_train = (n as f32 * 0.8) as usize;
    let train_data = &processed[..n_train];
    let val_data = &processed[n_train..];
    println!(
        "Train: {} | Validation: {}",
        train_data.len(),
        val_data.len()
    );

    // Create model
    println!("\nPhase 3: Creating Rosetta-Net with TCN...");
    let config = RosettaNetConfig {
        spectrogram_shape: (64, 64), // Smaller for memory efficiency
        latent_dim: 128,
        rosetta_dim: 45,
        num_classes,
        encoder_type: EncoderType::Hybrid,
        dropout_rate: 0.3,
        learning_rate: 0.001,
        rosetta_loss_weight: 10.0, // HEAVY weight on Rosetta regression
        classification_loss_weight: 1.0,
    };

    println!("Configuration:");
    println!("  Spectrogram shape: {:?}", config.spectrogram_shape);
    println!("  Latent dim: {}", config.latent_dim);
    println!("  Num classes: {}", config.num_classes);
    println!("  Encoder type: {:?}", config.encoder_type);
    println!(
        "  Rosetta loss weight: {} (HEAVY)",
        config.rosetta_loss_weight
    );
    println!(
        "  Classification loss weight: {}",
        config.classification_loss_weight
    );

    let mut model = RosettaNetWithTCN::new(config.clone());

    // Training loop (simplified - no actual weight updates without autograd)
    println!("\n{}", "═".repeat(60));
    println!("Phase 4: Training (Forward Pass Evaluation)");
    println!("{}", "═".repeat(60));
    println!("\nNote: Full training requires autograd (tch-rs/burn).");
    println!("Evaluating model with random weights as baseline...\n");

    // Evaluate on validation set
    let mut correct = 0usize;
    let mut total_rosetta_loss = 0.0f32;
    let mut total_class_loss = 0.0f32;

    for (spectrogram, features, label) in val_data.iter().take(500) {
        let (latent, rosetta_pred, logits) = model.forward(spectrogram);

        let target_class = label_to_idx.get(label).copied().unwrap_or(0);
        let rosetta_target = ndarray::Array1::from_vec(features.to_vec());

        let (_, rosetta_l, class_l) = compute_loss(
            &rosetta_pred,
            &rosetta_target,
            &logits,
            target_class,
            config.rosetta_loss_weight,
            config.classification_loss_weight,
        );

        total_rosetta_loss += rosetta_l;
        total_class_loss += class_l;

        // Check if prediction is correct
        let predicted = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if predicted == target_class {
            correct += 1;
        }
    }

    let n_val = val_data.len().min(500);
    let accuracy = correct as f32 / n_val as f32;
    let avg_rosetta_loss = total_rosetta_loss / n_val as f32;
    let avg_class_loss = total_class_loss / n_val as f32;

    // Results
    println!("{}", "═".repeat(60));
    println!("RESULTS: Rosetta-Net with TCN (Random Weights)");
    println!("{}", "═".repeat(60));
    println!("\n--- Loss Metrics ---");
    println!(
        "Avg Rosetta MSE: {:.4} (45D feature prediction)",
        avg_rosetta_loss
    );
    println!("Avg Cross-Entropy: {:.4} (classification)", avg_class_loss);

    println!("\n--- Classification Accuracy ---");
    println!("Validation accuracy: {:.2}%", accuracy * 100.0);

    println!("\n--- Comparison ---");
    println!("k-NN baseline:        38.56%");
    println!("Random Forest:        47.85%");
    println!("Rosetta-Net (random): {:.2}%", accuracy * 100.0);

    println!("\n{}", "─".repeat(60));
    println!("NEXT STEPS FOR TRAINING:");
    println!("{}", "─".repeat(60));
    println!("1. Add tch-rs or burn crate for autograd support");
    println!("2. Implement backpropagation through TCN layers");
    println!("3. Use Adam optimizer with learning rate 0.001");
    println!("4. Train for 50-100 epochs with early stopping");
    println!("5. Monitor: Rosetta MSE should decrease first,");
    println!("   then classification accuracy will improve");

    // Analyze temporal importance on a sample
    println!("\n--- Temporal vs Spectral Analysis ---");
    if let Some((spec, _, _)) = val_data.first() {
        let importance = model.analyze_temporal_importance(spec);
        println!(
            "Spectral contribution: {:.1}%",
            importance.spectral_contribution * 100.0
        );
        println!(
            "Temporal contribution: {:.1}%",
            importance.temporal_contribution * 100.0
        );
        println!("\nRandom Forest insight: duration_ms is 74% important");
        println!("TCN captures temporal dynamics that CNN misses");
    }

    println!("\nTotal time: {:.2}s", start_time.elapsed().as_secs_f64());

    // Save model configuration
    let config_path = "rosetta_net_config.json";
    let config_json = serde_json::to_string_pretty(&config)?;
    std::fs::write(config_path, config_json)?;
    println!("\nModel config saved to: {}", config_path);

    Ok(())
}
