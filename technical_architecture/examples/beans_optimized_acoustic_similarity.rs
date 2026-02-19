// BEANS-Zero Optimized Acoustic Similarity Assessment
//
// Uses library's ZooVoxFeatureExtractor with proper FFT for fast extraction.
// Combines parallel feature extraction with acoustic similarity-based typing.
//
// Usage:
//   cargo run --release --example beans_optimized_acoustic_similarity
//
// Performance: ~5-10x faster than naive DFT implementation

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use technical_architecture::{
    AcousticSimilarityEngine,
    SimilarityMetric,
    ZooVoxFeatureExtractor, // Uses proper FFT!
};

// ============================================================================
// Configuration
// ============================================================================

const SIMILARITY_THRESHOLD: f64 = 0.85;
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
    feature_dim: usize,
    total_time_sec: f64,
    throughput_samples_per_sec: f64,
    global_types: usize,
    type_entropy: f64,
    knn_accuracy: f64,
    knn_best_k: usize,
    avg_intra_type_similarity: f64,
    avg_inter_type_distance: f64,
    separation_ratio: f64,
    source_datasets: HashMap<String, usize>,
    task_types: HashMap<String, usize>,
}

// ============================================================================
// Parallel Feature Extraction using Library's FFT-based Extractor
// ============================================================================

fn extract_features_parallel(manifest: &Manifest, base_path: &Path) -> Vec<ExtractedFeatures> {
    let n_samples = manifest.samples.len();
    let sample_rate = manifest.resample_rate;

    println!(
        "Extracting {} samples using FFT-based ZooVoxFeatureExtractor...",
        n_samples
    );
    println!("  (Parallel with {} threads)", rayon::current_num_threads());
    println!();

    let processed = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    let all_features: Vec<ExtractedFeatures> = manifest
        .samples
        .par_iter()
        .filter_map(|sample| {
            // Create extractor for this thread (uses proper FFT)
            let mut extractor = ZooVoxFeatureExtractor::new(sample_rate);
            let audio_path = base_path.join(&sample.audio_file);

            match load_raw_audio(&audio_path, sample.n_samples) {
                Ok(audio) => {
                    let t0 = Instant::now();

                    // Convert f32 to f64 for the library
                    let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

                    match extractor.extract(&audio_f64) {
                        Ok(features_30d) => {
                            let extraction_time = t0.elapsed().as_secs_f64() * 1000.0;
                            let count = processed.fetch_add(1, Ordering::Relaxed);

                            if count % 5000 == 0 {
                                print!("\r  Progress: {}/{} samples", count + 1, n_samples);
                                use std::io::Write;
                                std::io::stdout().flush().ok();
                            }

                            Some(ExtractedFeatures {
                                sample_id: sample.id.clone(),
                                features: features_30d.to_vector().to_vec(),
                                duration_ms: sample.duration_ms,
                                labels: sample.labels.clone(),
                                extraction_time_ms: extraction_time,
                            })
                        }
                        Err(_) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                            None
                        }
                    }
                }
                Err(_) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .collect();

    let elapsed = start_time.elapsed();
    let n_processed = processed.load(Ordering::Relaxed);
    let n_failed = failed.load(Ordering::Relaxed);
    let throughput = n_processed as f64 / elapsed.as_secs_f64();

    println!(
        "\r  Progress: {}/{} samples processed",
        n_processed, n_samples
    );
    println!();
    println!("Extraction complete:");
    println!("  ├─ Processed: {} samples", n_processed);
    println!("  ├─ Failed: {} samples", n_failed);
    println!("  ├─ Time: {:.2}s", elapsed.as_secs_f64());
    println!("  └─ Throughput: {:.1} samples/sec", throughput);
    println!();

    all_features
}

fn load_raw_audio(path: &Path, expected_samples: usize) -> Result<Vec<f32>> {
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

// ============================================================================
// Type Discovery using Acoustic Similarity
// ============================================================================

fn build_global_types_streaming(
    features: &[ExtractedFeatures],
    similarity_threshold: f64,
) -> Vec<AcousticType> {
    if features.is_empty() {
        return Vec::new();
    }

    println!("Building global types using acoustic similarity engine...");
    println!("  (Streaming approach - memory efficient, no O(n²) matrix)");

    let n = features.len();

    // Create feature matrix
    let matrix = {
        let mut m = Array2::<f64>::zeros((n, FEATURE_DIM));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };

    // Create similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&matrix);

    // Streaming type assignment
    let mut types: Vec<AcousticType> = Vec::new();
    let max_types_to_check = 1000;

    for i in 0..n {
        let sample = matrix.row(i).to_owned();

        let mut best_type: Option<usize> = None;
        let mut best_sim = 0.0;

        let types_to_check = types.len().min(max_types_to_check);
        for type_idx in 0..types_to_check {
            let centroid = ndarray::Array1::from_vec(types[type_idx].centroid.clone());
            let sim = engine.similarity(&sample, &centroid);

            if sim >= similarity_threshold && sim > best_sim {
                best_sim = sim;
                best_type = Some(type_idx);
            }
        }

        if let Some(type_idx) = best_type {
            let n_in_type = types[type_idx].count + 1;
            types[type_idx].count = n_in_type;
            types[type_idx]
                .sample_ids
                .push(features[i].sample_id.clone());

            for (j, val) in features[i].features.iter().enumerate().take(FEATURE_DIM) {
                types[type_idx].centroid[j] +=
                    (val - types[type_idx].centroid[j]) / n_in_type as f64;
            }
        } else {
            types.push(AcousticType {
                type_id: format!("type_{}", types.len()),
                centroid: features[i].features.clone(),
                count: 1,
                sample_ids: vec![features[i].sample_id.clone()],
                avg_distance_to_centroid: 0.0,
            });
        }

        if (i + 1) % 10000 == 0 {
            println!("    {}/{} samples, {} types", i + 1, n, types.len());
        }
    }

    // Sort by count
    types.sort_by(|a, b| b.count.cmp(&a.count));

    // Compute avg distances for top types
    for t in types.iter_mut().take(100) {
        if t.count > 1 {
            let centroid = ndarray::Array1::from_vec(t.centroid.clone());
            let mut total_dist = 0.0;
            let mut count = 0;

            for sample_id in t.sample_ids.iter().take(50) {
                if let Some(f) = features.iter().find(|f| &f.sample_id == sample_id) {
                    let sample = ndarray::Array1::from_vec(f.features.clone());
                    total_dist += engine.distance(&centroid, &sample);
                    count += 1;
                }
            }

            if count > 0 {
                t.avg_distance_to_centroid = total_dist / count as f64;
            }
        }
    }

    println!("  Discovered {} types", types.len());
    types
}

// ============================================================================
// Statistics
// ============================================================================

fn compute_statistics(
    features: &[ExtractedFeatures],
    types: &[AcousticType],
) -> (f64, f64, f64, f64) {
    if features.is_empty() || types.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let total: usize = types.iter().map(|t| t.count).sum();
    let entropy = if total > 0 {
        types
            .iter()
            .map(|t| {
                let p = t.count as f64 / total as f64;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

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

    let mut intra_sim = 0.0;
    let mut intra_count = 0;

    for t in types.iter().take(50) {
        if t.count < 2 {
            continue;
        }

        let centroid = ndarray::Array1::from_vec(t.centroid.clone());

        for sample_id in t.sample_ids.iter().take(10) {
            if let Some(f) = features.iter().find(|f| &f.sample_id == sample_id) {
                let sample = ndarray::Array1::from_vec(f.features.clone());
                intra_sim += engine.similarity(&centroid, &sample);
                intra_count += 1;
            }
        }
    }

    let avg_intra = if intra_count > 0 {
        intra_sim / intra_count as f64
    } else {
        0.0
    };

    let mut inter_dist = 0.0;
    let mut inter_count = 0;

    for i in 0..types.len().min(50) {
        for j in (i + 1)..types.len().min(50) {
            let a = ndarray::Array1::from_vec(types[i].centroid.clone());
            let b = ndarray::Array1::from_vec(types[j].centroid.clone());
            inter_dist += engine.distance(&a, &b);
            inter_count += 1;
        }
    }

    let avg_inter = if inter_count > 0 {
        inter_dist / inter_count as f64
    } else {
        0.0
    };

    let separation = if avg_intra > 0.0 && avg_intra < 1.0 {
        avg_inter / (1.0 - avg_intra)
    } else {
        f64::INFINITY
    };

    (entropy, avg_intra, avg_inter, separation)
}

fn evaluate_knn(features: &[ExtractedFeatures]) -> (f64, usize) {
    let labels: Vec<String> = features
        .iter()
        .map(|f| {
            f.labels
                .get("source_dataset")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string())
        })
        .collect();

    let n = features.len();
    let eval_size = n.min(10000);
    let step = n / eval_size;

    let eval_indices: Vec<usize> = (0..n).step_by(step.max(1)).take(eval_size).collect();

    let eval_features = {
        let mut m = Array2::<f64>::zeros((eval_indices.len(), FEATURE_DIM));
        for (i, &idx) in eval_indices.iter().enumerate() {
            for (j, &val) in features[idx].features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };

    let eval_labels: Vec<String> = eval_indices
        .iter()
        .map(|&idx| labels[idx].clone())
        .collect();

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&eval_features);

    let n_folds = 5;
    let fold_size = eval_size / n_folds;

    let mut total_correct = 0;
    let mut total_tested = 0;

    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 {
            eval_size
        } else {
            (fold + 1) * fold_size
        };

        for i in test_start..test_end {
            let query = eval_features.row(i).to_owned();
            let true_label = &eval_labels[i];

            let mut distances: Vec<(usize, f64)> = (0..eval_size)
                .filter(|&j| j != i)
                .map(|j| {
                    let candidate = eval_features.row(j).to_owned();
                    (j, engine.distance(&query, &candidate))
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut votes: HashMap<String, usize> = HashMap::new();
            for (idx, _) in distances.iter().take(K_NEIGHBORS) {
                let label = &eval_labels[*idx];
                *votes.entry(label.clone()).or_insert(0) += 1;
            }

            let predicted = votes
                .into_iter()
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

fn analyze_labels(
    features: &[ExtractedFeatures],
) -> (HashMap<String, usize>, HashMap<String, usize>) {
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
    println!("║   BEANS-Zero Optimized Acoustic Similarity Assessment (FFT-based)         ║");
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
    println!("  ├─ Feature Dimension: {}D", FEATURE_DIM);
    println!("  ├─ Similarity Threshold: {:.2}", SIMILARITY_THRESHOLD);
    println!("  ├─ k-NN Neighbors: {}", K_NEIGHBORS);
    println!(
        "  ├─ Parallel: Rayon with {} threads",
        rayon::current_num_threads()
    );
    println!("  └─ Feature Extraction: ZooVoxFeatureExtractor (FFT-based)");
    println!();

    let total_start = Instant::now();

    // === Phase 1: Parallel Feature Extraction (FFT-based) ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 1: Parallel Feature Extraction (FFT-based)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let features = extract_features_parallel(&manifest, base_path);

    if features.is_empty() {
        anyhow::bail!("No features extracted!");
    }

    // === Phase 2: Type Discovery ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 2: Type Discovery (Acoustic Similarity Engine)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let global_types = build_global_types_streaming(&features, SIMILARITY_THRESHOLD);

    // === Phase 3: Statistics ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 3: Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (type_entropy, avg_intra, avg_inter, separation) =
        compute_statistics(&features, &global_types);

    println!("Type Discovery:");
    println!("  ├─ Global Types: {}", global_types.len());
    println!("  ├─ Type Entropy: {:.3} bits", type_entropy);
    println!();
    println!("Similarity Statistics:");
    println!("  ├─ Avg Intra-Type Similarity: {:.4}", avg_intra);
    println!("  ├─ Avg Inter-Type Distance: {:.4}", avg_inter);
    println!("  └─ Separation Ratio: {:.2}x", separation);
    println!();

    println!("Top 10 Types:");
    for (i, t) in global_types.iter().take(10).enumerate() {
        println!(
            "  {:2}. {} - {} samples, dist: {:.4}",
            i + 1,
            t.type_id,
            t.count,
            t.avg_distance_to_centroid
        );
    }
    println!();

    // === Phase 4: k-NN ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 4: k-NN Classification");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (knn_accuracy, knn_k) = evaluate_knn(&features);

    println!("k-NN ({}-NN): {:.2}% accuracy", knn_k, knn_accuracy * 100.0);
    println!();

    // === Phase 5: Labels ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 5: Label Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (source_datasets, task_types) = analyze_labels(&features);

    println!("Source Datasets:");
    for (source, count) in source_datasets.iter().take(10) {
        let pct = *count as f64 / features.len() as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", source, count, pct);
    }
    println!();

    println!("Task Types:");
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
    println!("Samples: {}", assessment.total_samples);
    println!("Features: {}D", assessment.feature_dim);
    println!();

    println!("Performance:");
    println!(
        "  ├─ Total time: {:.1}s ({:.1} min)",
        assessment.total_time_sec,
        assessment.total_time_sec / 60.0
    );
    println!(
        "  └─ Throughput: {:.1} samples/sec",
        assessment.throughput_samples_per_sec
    );
    println!();

    println!("Type Discovery:");
    println!("  ├─ Types: {}", assessment.global_types);
    println!("  ├─ Entropy: {:.3} bits", assessment.type_entropy);
    println!(
        "  ├─ Intra-sim: {:.4}",
        assessment.avg_intra_type_similarity
    );
    println!("  ├─ Inter-dist: {:.4}", assessment.avg_inter_type_distance);
    println!("  └─ Separation: {:.2}x", assessment.separation_ratio);
    println!();

    println!(
        "Classification: {}-NN @ {:.1}%",
        assessment.knn_best_k,
        assessment.knn_accuracy * 100.0
    );

    let competence = if assessment.knn_accuracy > 0.8 && assessment.separation_ratio > 2.0 {
        "EXCELLENT"
    } else if assessment.knn_accuracy > 0.7 && assessment.separation_ratio > 1.5 {
        "GOOD"
    } else if assessment.knn_accuracy > 0.6 {
        "MODERATE"
    } else {
        "NEEDS IMPROVEMENT"
    };

    println!();
    println!("Competence: {}", competence);

    // Save results
    let output_dir = PathBuf::from("beans_analysis");
    std::fs::create_dir_all(&output_dir)?;

    let results_path = output_dir.join("beans_optimized_acoustic_similarity_results.json");
    let file = File::create(&results_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &assessment)?;

    println!();
    println!("Results saved: {}", results_path.display());

    Ok(())
}
