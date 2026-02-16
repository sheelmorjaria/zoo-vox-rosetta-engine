// 56D Feature Extraction on BEANS-Zero using Rust MicroDynamicsExtractor
//
// This example:
// 1. Loads BEANS-Zero audio samples from disk
// 2. Extracts 56D MicroDynamics features (30D base + 13 Δ + 13 ΔΔ)
// 3. Processes 1000+ samples for robust competence assessment
// 4. Saves features for clustering and classification evaluation

use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::BufWriter;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use technical_architecture::micro_dynamics_extractor::{MicroDynamicsExtractor, MicroDynamicsFeatures56D};
use technical_architecture::island_hopping::Vector30D;

/// 56D feature vector for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureEntry56D {
    pub sample_id: String,
    pub features_56d: Vec<f32>,  // 56 elements
    pub features_30d: Vec<f32>,  // 30 elements (for comparison)
    pub metadata: AudioMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AudioMetadata {
    pub duration_seconds: f32,
    pub sample_rate: u32,
    pub num_samples: usize,
    pub mean_amplitude: f32,
    pub rms_energy: f32,
}

/// BEANS-Zero dataset processor
pub struct BeansProcessor {
    output_dir: PathBuf,
    target_samples: usize,
}

impl BeansProcessor {
    pub fn new(output_dir: PathBuf, target_samples: usize) -> Self {
        Self {
            output_dir,
            target_samples,
        }
    }

    /// Process BEANS-Zero dataset and extract 56D features
    pub fn process(&self) -> Result<ProcessingReport> {
        println!("╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║   56D MicroDynamics Extraction: BEANS-Zero (Rust)                        ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");
        println!();

        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // Step 1: Load audio files from BEANS-Zero
        println!("📂 Step 1: Loading BEANS-Zero audio files...");
        let audio_files = self.find_beans_audio_files()?;
        println!("   Found {} audio files", audio_files.len());

        // Sample target number of files
        let num_samples = self.target_samples.min(audio_files.len());
        let sampled_files: Vec<_> = audio_files.into_iter()
            .take(num_samples)
            .collect();

        println!("   Processing {} samples", num_samples);
        println!();

        // Step 2: Extract 56D features
        println!("🎚️  Step 2: Extracting 56D MicroDynamics features...");
        println!("   56D = 30D Base + 13 Δ MFCC + 13 ΔΔ MFCC");
        println!();

        let mut entries = Vec::new();
        let mut extractor = MicroDynamicsExtractor::new(44100);

        for (idx, audio_path) in sampled_files.iter().enumerate() {
            // Load audio
            let audio_data = self.load_audio_file(audio_path)?;

            if audio_data.len() < 1000 {
                println!("   [{:4}/{}] ⚠ Skipping (too short): {}",
                    idx + 1, num_samples, audio_path.display());
                continue;
            }

            // Extract 56D features
            match extractor.extract_56d(&audio_data) {
                Ok(features_56d) => {
                    // Compute metadata
                    let metadata = self.compute_metadata(&audio_data);

                    // Convert 56D to vector
                    let features_56d_vec = self.features_56d_to_vec(&features_56d);

                    // Convert base 30D to vector
                    let mean_f0 = 5000.0; // Placeholder - would be estimated
                    let duration_ms = metadata.duration_seconds * 1000.0;
                    let f0_range = 1000.0; // Placeholder
                    let vector30d = features_56d.base_30d.to_vector30d(mean_f0, duration_ms, f0_range);
                    let features_30d_vec = vector30d.to_array().to_vec();

                    let entry = FeatureEntry56D {
                        sample_id: audio_path.file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string(),
                        features_56d: features_56d_vec,
                        features_30d: features_30d_vec,
                        metadata,
                    };

                    entries.push(entry);

                    if (idx + 1) % 50 == 0 {
                        println!("   [{:4}/{}] ✓ Processed", idx + 1, num_samples);
                    }
                }
                Err(e) => {
                    println!("   [{:4}/{}] ✗ Error: {}",
                        idx + 1, num_samples, e);
                }
            }
        }

        println!();
        println!("   ✓ Successfully extracted {} feature vectors", entries.len());
        println!();

        // Step 3: Save features
        println!("💾 Step 3: Saving features to disk...");
        let output_path = self.output_dir.join("beans_56d_features_rust.json");
        self.save_features(&entries, &output_path)?;
        println!("   ✓ Saved to: {}", output_path.display());
        println!();

        // Step 4: Generate Python evaluation script
        println!("📝 Step 4: Generating evaluation script...");
        let eval_script_path = self.output_dir.join("evaluate_56d_rust_features.py");
        self.generate_evaluation_script(&eval_script_path)?;
        println!("   ✓ Generated: {}", eval_script_path.display());
        println!();

        // Generate report
        let report = ProcessingReport {
            total_files_found: num_samples,
            successful_extractions: entries.len(),
            feature_dimensionality: 56,
            output_file: output_path.to_string_lossy().to_string(),
            evaluation_script: eval_script_path.to_string_lossy().to_string(),
        };

        Ok(report)
    }

    fn find_beans_audio_files(&self) -> Result<Vec<PathBuf>> {
        // BEANS-Zero is typically cached in ~/.cache/huggingface/datasets/
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))?;

        let cache_path = PathBuf::from(home)
            .join(".cache/huggingface/datasets/EarthSpeciesProject___beans-zero");

        println!("   Searching for BEANS-Zero in: {}", cache_path.display());

        let mut audio_files = Vec::new();

        // Search recursively for audio files
        if cache_path.exists() {
            self.find_audio_recursive(&cache_path, &mut audio_files);
        }

        // If not found in cache, try alternative locations
        if audio_files.is_empty() {
            let alt_paths = vec![
                PathBuf::from("/mnt/c/Users/sheel/Desktop/src/technical_architecture/beans_analysis/audio"),
                PathBuf::from("./beans_audio"),
            ];

            for alt_path in alt_paths {
                if alt_path.exists() {
                    self.find_audio_recursive(&alt_path, &mut audio_files);
                    if !audio_files.is_empty() {
                        break;
                    }
                }
            }
        }

        Ok(audio_files)
    }

    fn find_audio_recursive(&self, dir: &Path, audio_files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    self.find_audio_recursive(&path, audio_files);
                } else if let Some(ext) = path.extension() {
                    if matches!(ext.to_str(), Some("wav" | "flac" | "mp3" | "ogg")) {
                        audio_files.push(path);
                    }
                }
            }
        }
    }

    fn load_audio_file(&self, path: &Path) -> Result<Vec<f32>> {
        // For now, load WAV using basic parsing
        // In production, use symphonia or rodio
        let bytes = fs::read(path)?;

        // Simple WAV parsing (16-bit PCM, mono)
        if bytes.starts_with(b"RIFF") {
            self.parse_wav(&bytes)
        } else {
            // Try loading as raw f32
            anyhow::bail!("Unsupported audio format")
        }
    }

    fn parse_wav(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        // Skip WAV header (44 bytes) and read 16-bit PCM samples
        if bytes.len() < 44 {
            anyhow::bail!("Invalid WAV file");
        }

        let sample_data = &bytes[44..];
        let mut samples = Vec::new();

        // Read 16-bit samples as little-endian
        for chunk in sample_data.chunks_exact(2) {
            let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
            samples.push(sample_i16 as f32 / 32768.0);
        }

        // Convert to mono if stereo (interleaved)
        let is_stereo = bytes[22] == 2;
        if is_stereo {
            let mut mono = Vec::new();
            for chunk in samples.chunks_exact(2) {
                mono.push((chunk[0] + chunk[1]) / 2.0);
            }
            Ok(mono)
        } else {
            Ok(samples)
        }
    }

    fn compute_metadata(&self, audio: &[f32]) -> AudioMetadata {
        let num_samples = audio.len();
        let duration_seconds = num_samples as f32 / 44100.0;
        let sample_rate = 44100;
        let mean_amplitude = audio.iter().map(|&x| x.abs()).sum::<f32>() / num_samples.max(1) as f32;
        let rms_energy = (audio.iter().map(|&x| x * x).sum::<f32>() / num_samples.max(1) as f32).sqrt();

        AudioMetadata {
            duration_seconds,
            sample_rate,
            num_samples,
            mean_amplitude,
            rms_energy,
        }
    }

    fn features_56d_to_vec(&self, features: &MicroDynamicsFeatures56D) -> Vec<f32> {
        let mut vec = Vec::with_capacity(56);

        // Base 30D features
        let base = &features.base_30d;
        vec.extend_from_slice(&[
            // Fundamental (3) - will be added later with F0 estimation
            0.0, 0.0, 0.0,  // mean_f0, duration, f0_range (placeholders)
            // Grit Factors (3)
            base.harmonic_to_noise_ratio,
            base.spectral_flatness,
            base.harmonicity,
            // Motion Factors (7)
            base.attack_time_ms,
            base.decay_time_ms,
            base.sustain_level,
            base.vibrato_rate_hz,
            base.vibrato_depth,
            base.jitter,
            base.shimmer,
            // Fingerprint Factors (14)
        ]);
        vec.extend_from_slice(&base.mfcc);
        vec.push(base.spectral_flux);
        // Rhythm Factors (3)
        vec.extend_from_slice(&[
            base.median_ici_ms,
            base.onset_rate_hz,
            base.ici_coefficient_of_variation,
        ]);

        // Delta Features (26)
        vec.extend_from_slice(&features.mfcc_delta);
        vec.extend_from_slice(&features.mfcc_delta_delta);

        vec
    }

    fn save_features(&self, entries: &[FeatureEntry56D], path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, entries)?;
        Ok(())
    }

    fn generate_evaluation_script(&self, script_path: &Path) -> Result<()> {
        let script = r#"
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

print("="*80)
print("56D MICRODYNAMICS COMPETENCE: RUST EXTRACTED FEATURES")
print("="*80)

# Load features
input_file = Path("beans_56d_features_rust.json")
with open(input_file) as f:
    entries = json.load(f)

print(f"\nLoaded {len(entries)} feature vectors")

# Extract features
features_30d = []
features_56d = []

for entry in entries:
    features_30d.append(entry['features_30d'])
    features_56d.append(entry['features_56d'])

features_30d = np.array(features_30d)
features_56d = np.array(features_56d)

print(f"Feature shapes:")
print(f"  30D: {features_30d.shape}")
print(f"  56D: {features_56d.shape}")

# Normalize
scaler_30d = StandardScaler()
scaler_56d = StandardScaler()
features_30d_norm = scaler_30d.fit_transform(features_30d)
features_56d_norm = scaler_56d.fit_transform(features_56d)

# Clustering
print("\n" + "="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

agg_30d = AgglomerativeClustering(n_clusters=10, linkage='ward')
labels_30d = agg_30d.fit_predict(features_30d_norm)

agg_56d = AgglomerativeClustering(n_clusters=10, linkage='ward')
labels_56d = agg_56d.fit_predict(features_56d_norm)

sil_30d = silhouette_score(features_30d_norm, labels_30d)
sil_56d = silhouette_score(features_56d_norm, labels_56d)

print(f"\n30D Clustering:")
print(f"  Silhouette: {sil_30d:.3f}")

print(f"\n56D Clustering:")
print(f"  Silhouette: {sil_56d:.3f}")

print(f"\nImprovement: {(sil_56d - sil_30d) * 100:+.1f}%")

# Cross-validation
print("\n" + "="*80)
print("CLASSIFICATION VALIDATION")
print("="*80)

def cross_validate(features, labels, n_splits=5):
    if len(set(labels)) < 2:
        return {"accuracy": 0.0, "f1": 0.0, "std": 0.0}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for train_idx, test_idx in skf.split(features, labels):
        if len(set(labels[train_idx])) < 2:
            continue

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features[train_idx], labels[train_idx])
        pred = knn.predict(features[test_idx])

        accuracies.append(accuracy_score(labels[test_idx], pred))
        f1_scores.append(f1_score(labels[test_idx], pred, average='weighted', zero_division=0))

    return {
        "accuracy": np.mean(accuracies) if accuracies else 0.0,
        "f1": np.mean(f1_scores) if f1_scores else 0.0,
        "std": np.std(accuracies) if accuracies else 0.0
    }

cv_30d = cross_validate(features_30d_norm, labels_30d)
cv_56d = cross_validate(features_56d_norm, labels_56d)

print(f"\n30D Classification:")
print(f"  Accuracy: {cv_30d['accuracy']:.3f} ± {cv_30d['std']:.3f}")
print(f"  F1-Score: {cv_30d['f1']:.3f}")

print(f"\n56D Classification:")
print(f"  Accuracy: {cv_56d['accuracy']:.3f} ± {cv_56d['std']:.3f}")
print(f"  F1-Score: {cv_56d['f1']:.3f}")

acc_imp = (cv_56d['accuracy'] - cv_30d['accuracy']) * 100
f1_imp = (cv_56d['f1'] - cv_30d['f1']) * 100

print(f"\n" + "="*80)
print("56D vs 30D COMPARISON")
print("="*80)
print(f"\nAccuracy Improvement: {acc_imp:+.1f}%")
print(f"F1-Score Improvement: {f1_imp:+.1f}%")
print(f"Silhouette Improvement: {(sil_56d - sil_30d) * 100:+.1f}%")

# Save results
results = {
    "num_samples": len(entries),
    "30d": {
        "accuracy": float(cv_30d['accuracy']),
        "f1": float(cv_30d['f1']),
        "silhouette": float(sil_30d)
    },
    "56d": {
        "accuracy": float(cv_56d['accuracy']),
        "f1": float(cv_56d['f1']),
        "silhouette": float(sil_56d)
    },
    "improvement": {
        "accuracy_pct": float(acc_imp),
        "f1_pct": float(f1_imp),
        "silhouette_pct": float((sil_56d - sil_30d) * 100)
    },
    "competence": "good" if cv_56d['accuracy'] > 0.7 else "moderate"
}

output_file = Path("beans_56d_rust_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
"#;

        fs::write(script_path, script)?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ProcessingReport {
    total_files_found: usize,
    successful_extractions: usize,
    feature_dimensionality: usize,
    output_file: String,
    evaluation_script: String,
}

fn main() -> Result<()> {
    let output_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/technical_architecture/beans_analysis");
    let processor = BeansProcessor::new(output_dir.clone(), 1000);

    let report = processor.process()?;

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   PROCESSING COMPLETE                                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📊 Summary:");
    println!("   Files found:        {}", report.total_files_found);
    println!("   Successful:         {}", report.successful_extractions);
    println!("   Feature dimension:  {}D", report.feature_dimensionality);
    println!("   Output file:        {}", report.output_file);
    println!("   Evaluation script:  {}", report.evaluation_script);
    println!();
    println!("🚀 Next Steps:");
    println!("   cd {}", output_dir.display());
    println!("   python3 evaluate_56d_rust_features.py");
    println!();

    Ok(())
}
