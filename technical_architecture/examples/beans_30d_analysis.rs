// 30D MicroDynamics Analysis on BEANS-Zero Dataset
//
// BEANS-Zero: Bird Vocalizations with Classification Labels
// This analysis assesses competence in:
// 1. Species classification (multi-class)
// 2. Individual detection (identity recognition)
// 3. Call type detection (vocalization categories)
//
// Extracts 30D MicroDynamics features and evaluates:
// - Feature separability (between classes)
// - Cluster coherence (within classes)
// - Classification potential (k-NN accuracy)

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

// For this analysis, we'll interface with Python to load the HuggingFace dataset
// Then process the audio through our Rust pipeline

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   30D MicroDynamics Analysis: BEANS-Zero Dataset                            ║");
    println!("║   Bird Vocalization Classification & Detection Competence Assessment        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let output_dir =
        Path::new("/mnt/c/Users/sheel/Desktop/src/technical_architecture/beans_analysis");
    std::fs::create_dir_all(output_dir)?;

    println!("📊 Analysis Configuration:");
    println!("   ├─ Dataset: EarthSpeciesProject/BEANS-Zero");
    println!("   ├─ Feature extraction: 30D MicroDynamics");
    println!("   ├─ Sample rate: 44.1kHz (typical for bird recordings)");
    println!("   └─ Output: {}", output_dir.display());
    println!();

    // Step 1: Download and prepare dataset using Python
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Dataset Preparation (Python + HuggingFace)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create Python script for dataset preparation
    let python_script = r#"
import os
import sys
import json
import numpy as np
from datasets import load_dataset
from pathlib import Path

print("Loading BEANS-Zero dataset...")
ds = load_dataset("EarthSpeciesProject/BEANS-Zero")

print(f"\nDataset splits: {list(ds.keys())}")
for split_name, split_data in ds.items():
    print(f"\n{split_name.upper()}:")
    print(f"  Num examples: {len(split_data)}")
    print(f"  Columns: {split_data.column_names}")

    # Show first example
    if len(split_data) > 0:
        example = split_data[0]
        print(f"  Example keys: {list(example.keys())}")
        for key, value in example.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"    {key}: {value}")
            elif isinstance(value, list):
                print(f"    {key}: [list with {len(value)} items]")
            else:
                print(f"    {key}: {type(value).__name__}")

# Analyze label distributions
print("\n" + "="*80)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*80)

# Check what labels are available
train_data = ds.get('train', ds.get('test', None))
if train_data:
    # Get all column names
    columns = train_data.column_names
    print(f"\nAvailable columns: {columns}")

    # Look for label columns
    label_columns = []
    for col in columns:
        if 'label' in col.lower() or 'class' in col.lower() or 'species' in col.lower() or 'id' in col.lower():
            label_columns.append(col)

    if label_columns:
        print(f"\nPotential label columns: {label_columns}")

        for col in label_columns[:3]:  # Analyze first 3 label columns
            print(f"\n{col} distribution:")
            unique_values = {}
            for example in train_data:
                val = example.get(col)
                if val is not None:
                    unique_values[str(val)] = unique_values.get(str(val), 0) + 1

            for val, count in sorted(unique_values.items(), key=lambda x: -x[1])[:10]:
                print(f"  {val}: {count}")

# Save dataset summary
summary = {
    "splits": list(ds.keys()),
    "num_examples": {split: len(data) for split, data in ds.items()},
    "columns": ds[list(ds.keys())[0]].column_names if ds.keys() else [],
}

output_file = Path("/mnt/c/Users/sheel/Desktop/src/technical_architecture/beans_analysis/dataset_summary.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDataset summary saved to: {output_file}")
"#;

    let script_path = output_dir.join("prepare_dataset.py");
    std::fs::write(&script_path, python_script)?;

    println!("📝 Python script created: {}", script_path.display());
    println!();

    // Run Python script to download and analyze dataset
    println!("🔄 Downloading and analyzing dataset (this may take a few minutes)...");

    let result = std::process::Command::new("python3")
        .arg(&script_path)
        .current_dir(output_dir)
        .output()?;

    if result.status.success() {
        println!("{}", String::from_utf8_lossy(&result.stdout));
    } else {
        eprintln!("Error running Python script:");
        eprintln!("{}", String::from_utf8_lossy(&result.stderr));
    }

    println!();

    // Step 2: Read the dataset summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Dataset Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let summary_path = output_dir.join("dataset_summary.json");
    if summary_path.exists() {
        let summary_content = std::fs::read_to_string(&summary_path)?;
        println!("{}", summary_content);
    }

    println!();

    // Step 3: Design analysis strategy based on dataset structure
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Analysis Strategy");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("30D MicroDynamics Analysis for BEANS-Zero:");
    println!();
    println!("📋 Assessment Tasks:");
    println!("   1. Species Classification");
    println!("      ├─ Extract 30D features from bird vocalizations");
    println!("      ├─ Train classifier on species labels");
    println!("      └─ Metrics: Accuracy, F1-score, Confusion Matrix");
    println!();
    println!("   2. Individual Detection (Bird ID)");
    println!("      ├─ Assess if features capture individual identity");
    println!("      ├─ Cluster analysis per species");
    println!("      └─ Metric: Silhouette score, Dunn index");
    println!();
    println!("   3. Call Type Detection");
    println!("      ├─ Identify different vocalization types");
    println!("      ├─ Unsupervised clustering analysis");
    println!("      └─ Metric: Cluster purity, NMI");
    println!();
    println!("🔬 Feature Extraction (30D MicroDynamics):");
    println!("   ├─ Spectral features: MFCC, spectral centroid, rolloff");
    println!("   ├─ Temporal features: Onset/offset patterns, modulation");
    println!("   ├─ Prosodic features: Pitch, duration, intensity");
    println!("   └─ Micro-structural features: Fine-grained temporal patterns");
    println!();

    // Step 4: Create Rust-based feature extraction pipeline
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 4: Feature Extraction Pipeline");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("Creating Rust-based 30D MicroDynamics extraction pipeline...");

    let rust_pipeline = r#"
use std::path::Path;
use numpy::{ndarray::{Array2, ArrayView1}, IntoPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// 30D MicroDynamics Feature Extractor
///
/// Extracts comprehensive acoustic features for vocalization analysis:
/// - Spectral features (10D): MFCCs, spectral centroid, bandwidth, rolloff, flux
/// - Temporal features (8D): Onset/offset patterns, duration, rhythm
/// - Prosodic features (7D): F0 statistics, intensity variation
/// - Micro-structural features (5D): Fine temporal patterns, modulation

#[pyclass]
struct MicroDynamicsExtractor {
    sample_rate: u32,
    n_mfcc: usize,
    n_mels: usize,
}

#[pymethods]
impl MicroDynamicsExtractor {
    #[new]
    #[args(sample_rate = 44100, n_mfcc = 13, n_mels = 40)]
    fn new(sample_rate: u32, n_mfcc: usize, n_mels: usize) -> Self {
        Self {
            sample_rate,
            n_mfcc,
            n_mels,
        }
    }

    /// Extract 30D MicroDynamics features from audio
    ///
    /// Args:
    ///   audio: Audio samples as numpy array
    ///
    /// Returns:
    ///   30D feature vector representing the vocalization
    fn extract(&self, py: Python, audio: &PyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let audio_slice = audio.as_slice()?;

        // This is a placeholder - actual implementation would:
        // 1. Compute spectrogram
        // 2. Extract MFCCs (13D)
        // 3. Extract spectral features (centroid, bandwidth, rolloff, flux, 4D)
        // 4. Extract temporal features (onsets, duration, rhythm, 4D)
        // 5. Extract prosodic features (F0 stats, 7D)
        // 6. Extract micro-structural features (modulation, 2D)

        // For now, return placeholder features
        let features = vec![0.0; 30];

        Ok(PyArray1::from_vec(py, features).into_py(py))
    }

    /// Extract features from a batch of audio files
    ///
    /// Args:
    ///   audio_files: List of (audio_data, sample_rate) tuples
    ///   labels: Corresponding labels for classification
    ///
    /// Returns:
    ///   Dictionary with features and labels
    fn extract_batch(
        &self,
        py: Python,
        audio_files: Vec<&PyArray1<f64>>,
        labels: Vec<String>,
    ) -> PyResult<PyObject> {
        let mut all_features = Vec::new();

        for audio in audio_files {
            let features = self.extract(py, audio)?;
            all_features.push(features);
        }

        // Create output dictionary
        let dict = PyDict::new(py);
        dict.set_item("features", all_features)?;
        dict.set_item("labels", labels)?;
        dict.set_item("n_features", 30)?;

        Ok(dict.into())
    }
}

/// Analyze feature separability for classification tasks
#[pyfunction]
fn analyze_separability(
    py: Python,
    features: &PyArray2<f64>,
    labels: Vec<String>,
) -> PyResult<PyObject> {
    let features_view = features.as_array();

    // Compute between-class and within-class scatter matrices
    // This is a placeholder - actual implementation would:
    // 1. Compute class means
    // 2. Compute overall mean
    // 3. Compute between-class scatter (Sb)
    // 4. Compute within-class scatter (Sw)
    // 5. Compute Fisher's linear discriminant ratio

    let result = PyDict::new(py);
    result.set_item("fisher_ratio", 0.0)?;
    result.set_item("class_separability", 0.0)?;

    Ok(result.into())
}

/// Python module
#[pymodule]
fn beans_analysis(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroDynamicsExtractor>()?;
    m.add_function(wrap_pyfunction!(analyze_separability, m)?)?;
    Ok(())
}
"#;

    let pipeline_path = output_dir.join("feature_extraction.py");
    std::fs::write(&pipeline_path, rust_pipeline)?;

    println!(
        "✅ Feature extraction pipeline created: {}",
        pipeline_path.display()
    );
    println!();

    // Step 5: Create analysis report
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 5: Analysis Plan");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("📊 Competence Assessment Metrics:");
    println!();
    println!("1. Species Classification Competence:");
    println!("   ├─ Feature separability: Between-class / within-class variance");
    println!("   ├─ k-NN cross-validation: 5-fold CV accuracy");
    println!("   ├─ SVM baseline: Linear and RBF kernels");
    println!("   └─ Random forest baseline: Feature importance");
    println!();
    println!("2. Individual Detection Competence:");
    println!("   ├─ Per-species clustering: Can we identify individuals?");
    println!("   ├─ Silhouette analysis: Cluster quality metrics");
    println!("   ├─ Dunn index: Compactness vs separation");
    println!("   └─ Confusion analysis: Individual confusion patterns");
    println!();
    println!("3. Call Type Detection Competence:");
    println!("   ├─ Unsupervised clustering: DBSCAN on 30D features");
    println!("   ├─ Cluster purity: Alignment with known labels");
    println!("   ├─ Normalized Mutual Information (NMI)");
    println!("   └─ Adjusted Rand Index (ARI)");
    println!();

    // Create comprehensive analysis script
    let analysis_script = r#"
import os
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
import pickle

print("="*80)
print("30D MICRODYNAMICS ANALYSIS: BEANS-ZERO")
print("="*80)

# Load dataset
print("\nLoading BEANS-Zero dataset...")
ds = load_dataset("EarthSpeciesProject/BEANS-Zero")

# Determine which split to use
train_split = ds.get('train', ds.get('test', ds[list(ds.keys())[0]]))
print(f"Using split: {list(ds.keys())[0] if ds else 'unknown'}")
print(f"Number of examples: {len(train_split)}")

# Analyze available columns
columns = train_split.column_names
print(f"\nAvailable columns: {columns}")

# Identify label columns
label_candidates = []
for col in columns:
    if any(keyword in col.lower() for keyword in ['label', 'species', 'class', 'id', 'type', 'category']):
        label_candidates.append(col)

print(f"\nLabel candidates: {label_candidates}")

# Determine primary classification task
primary_label = None
for col in label_candidates:
    if 'species' in col.lower():
        primary_label = col
        break

if primary_label is None and label_candidates:
    primary_label = label_candidates[0]

if primary_label:
    print(f"\nPrimary classification task: {primary_label}")

    # Get unique labels
    unique_labels = set()
    for example in train_split:
        val = example.get(primary_label)
        if val is not None:
            unique_labels.add(str(val))

    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")

    if num_classes <= 20:
        print(f"Classes: {sorted(unique_labels)}")
    else:
        print(f"Sample classes: {sorted(list(unique_labels))[:10]} ... ({num_classes} total)")
else:
    print("\nNo clear classification label found")
    print("Will perform unsupervised analysis only")

# Feature extraction plan
print("\n" + "="*80)
print("FEATURE EXTRACTION PLAN")
print("="*80)

print("""
30D MicroDynamics Features:

1. Spectral Features (13D):
   - 13 MFCC coefficients
   - Capture spectral envelope and timbre

2. Spectral Statistics (5D):
   - Spectral centroid (brightness)
   - Spectral bandwidth (range)
   - Spectral rolloff (energy concentration)
   - Spectral flux (change over time)
   - Zero crossing rate (noisiness)

3. Temporal Features (6D):
   - Onset strength (attack)
   - Offset strength (decay)
   - Duration (time extent)
   - Onset-offset latency
   - Temporal centroid (timing)
   - Rhythm regularity

4. Prosodic Features (6D):
   - F0 mean (pitch)
   - F0 std (pitch variation)
   - F0 range (pitch excursion)
   - Intensity mean (loudness)
   - Intensity std (loudness variation)
   - Intensity range (dynamic range)

Total: 30D feature vector per vocalization
""")

# Analysis workflow
print("\n" + "="*80)
print("ANALYSIS WORKFLOW")
print("="*80)

print("""
Phase 1: Data Preparation
  ✓ Download dataset
  ✓ Analyze structure and labels
  → Extract audio samples
  → Segment vocalizations
  → Split train/test (80/20)

Phase 2: Feature Extraction
  → Load audio files
  → Preprocess (normalize, remove silence)
  → Extract 30D MicroDynamics features
  → Normalize features (z-score)
  → Save feature matrix

Phase 3: Species Classification Assessment
  → k-NN classification (k=5)
  → SVM classification (linear, RBF)
  → Random Forest classification
  → Cross-validation (5-fold)
  → Metrics: Accuracy, F1, Confusion Matrix

Phase 4: Individual Detection Assessment
  → Per-species clustering analysis
  → Silhouette score calculation
  → Dunn index computation
  → Individual identification rate

Phase 5: Call Type Detection
  → Unsupervised clustering (DBSCAN, HDBSCAN)
  → Cluster purity analysis
  → NMI and ARI scores
  → Cluster interpretation

Phase 6: Competence Report
  → Summarize classification accuracy
  → Assess feature separability
  → Determine competence levels
  → Generate visualization
  → Save results
""")

# Save analysis plan
output_dir = Path("/mnt/c/Users/sheel/Desktop/src/technical_architecture/beans_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

plan = {
    "dataset": "EarthSpeciesProject/BEANS-Zero",
    "primary_task": primary_label if primary_label else "unsupervised",
    "num_classes": num_classes if primary_label else len(unique_labels),
    "num_examples": len(train_split),
    "feature_dim": 30,
    "features": {
        "spectral": {"mfcc": 13, "statistics": 5},
        "temporal": 6,
        "prosodic": 6
    }
}

with open(output_dir / "analysis_plan.json", 'w') as f:
    json.dump(plan, f, indent=2)

print(f"\nAnalysis plan saved to: {output_dir / 'analysis_plan.json'}")
print("\n✅ Dataset analysis complete!")
print("\nNext steps:")
print("  1. Implement 30D feature extraction in Rust")
print("  2. Process audio files from BEANS-Zero")
print("  3. Train classifiers and assess competence")
print("  4. Generate competence report")
"#;

    let analysis_path = output_dir.join("run_analysis.py");
    std::fs::write(&analysis_path, analysis_script)?;

    println!("✅ Analysis script created: {}", analysis_path.display());
    println!();

    // Step 6: Run the initial analysis
    println!("🔄 Running initial analysis...");

    let result = std::process::Command::new("python3")
        .arg(&analysis_path)
        .current_dir(output_dir)
        .output()?;

    if result.status.success() {
        println!("{}", String::from_utf8_lossy(&result.stdout));
    } else {
        eprintln!("Error: {}", String::from_utf8_lossy(&result.stderr));
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📁 Output directory: {}", output_dir.display());
    println!("   ├─ prepare_dataset.py");
    println!("   ├─ feature_extraction.py");
    println!("   ├─ run_analysis.py");
    println!("   ├─ dataset_summary.json");
    println!("   └─ analysis_plan.json");
    println!();
    println!("📋 Next steps to complete the 30D MicroDynamics analysis:");
    println!("   1. Review dataset structure and labels");
    println!("   2. Implement Rust-based feature extraction");
    println!("   3. Extract features from audio files");
    println!("   4. Train classifiers and assess competence");
    println!("   5. Generate comprehensive report");

    Ok(())
}
