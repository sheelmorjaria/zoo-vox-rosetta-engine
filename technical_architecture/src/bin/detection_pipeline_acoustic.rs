//! Bioacoustic Detection Pipeline with Acoustic Grouping
//!
//! This pipeline uses 13 acoustic specialists instead of 8 taxonomic specialists.
//! Acoustic grouping provides better feature coherence within each specialist.

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::acoustic_router::{
    build_canonical_label_map, map_species_to_acoustic_group, normalize_label, AcousticGroup, FEATURE_DIM,
};
use technical_architecture::classical_ml::RandomForestClassifier;

#[derive(Parser, Debug)]
#[command(name = "detection_pipeline_acoustic", version = "1.0")]
struct Args {
    #[arg(short, long, default_value = "detections_acoustic.json")]
    output: String,

    #[arg(short, long, default_value = "1.5")]
    threshold: f32,

    #[arg(long, default_value = "0")]
    max_segments: usize,

    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionPayload {
    pub features_112d: Vec<f32>,
    pub start_time_ms: f64,
    pub end_time_ms: f64,
    pub source_file: String,
    pub segment_idx: usize,
    pub true_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub start_s: f64,
    pub end_s: f64,
    pub duration_s: f64,
    pub species: String,
    pub confidence: f32,
    pub acoustic_group: String,
    pub source_file: String,
    pub true_label: String,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStats {
    pub total_segments: usize,
    pub positive_detections: usize,
    pub rejected_by_threshold: usize,
    pub correct_detections: usize,
    pub accuracy: f64,
    pub avg_inference_time_us: f64,
    pub acoustic_group_distribution: HashMap<String, usize>,
}

/// Acoustic Hierarchical Router
struct AcousticHierarchicalRouter {
    specialists: HashMap<AcousticGroup, RandomForestClassifier>,
    canonical_map: HashMap<String, String>,
    threshold: f32,
}

impl AcousticHierarchicalRouter {
    fn new(threshold: f32) -> Self {
        Self {
            specialists: HashMap::new(),
            canonical_map: build_canonical_label_map(),
            threshold,
        }
    }

    fn load(&mut self, group: AcousticGroup, path: &Path) -> Result<()> {
        let start = Instant::now();

        // Try bincode first (preferred)
        let bincode_path = path.with_extension("bincode");
        if bincode_path.exists() {
            let file = fs::File::open(&bincode_path)?;
            let reader = BufReader::new(file);
            let model: RandomForestClassifier = bincode::deserialize_from(reader)?;
            println!(
                "  {:?}: {} trees, {} classes (bincode, {:.2}s)",
                group,
                model.n_trees(),
                model.n_classes(),
                start.elapsed().as_secs_f32()
            );
            self.specialists.insert(group, model);
            return Ok(());
        }

        // Fallback to JSON
        if path.exists() {
            let data = fs::read_to_string(path)?;
            let model: RandomForestClassifier = serde_json::from_str(&data)?;
            println!(
                "  {:?}: {} trees, {} classes (JSON, {:.2}s)",
                group,
                model.n_trees(),
                model.n_classes(),
                start.elapsed().as_secs_f32()
            );
            self.specialists.insert(group, model);
        }

        Ok(())
    }

    fn detect(&self, payload: &DetectionPayload) -> Option<(Detection, u64)> {
        let acoustic_group = map_species_to_acoustic_group(&payload.true_label);
        let model = self.specialists.get(&acoustic_group)?;
        let start = Instant::now();

        let features = ndarray::Array1::from_vec(payload.features_112d.clone());
        let pred_idx = model.predict(&features);
        let proba = model.predict_proba(&features);
        let confidence = *proba.get(pred_idx).unwrap_or(&0.0);

        if confidence < self.threshold {
            return None;
        }

        let species_map = model.idx_to_label();
        let species = match species_map.get(&pred_idx) {
            Some(s) => s.clone(),
            None => return None,
        };
        let species_canonical = normalize_label(&species, &self.canonical_map);
        let true_canonical = normalize_label(&payload.true_label, &self.canonical_map);

        let inference_time_us = start.elapsed().as_micros() as u64;

        Some((
            Detection {
                start_s: payload.start_time_ms / 1000.0,
                end_s: payload.end_time_ms / 1000.0,
                duration_s: (payload.end_time_ms - payload.start_time_ms) / 1000.0,
                species: species_canonical.clone(),
                confidence,
                acoustic_group: format!("{:?}", acoustic_group),
                source_file: payload.source_file.clone(),
                true_label: true_canonical.clone(),
                correct: species_canonical == true_canonical,
            },
            inference_time_us,
        ))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Bioacoustic Detection Pipeline (Acoustic Grouping)                      ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!(
        "\nConfig: threshold={:.2}, max_segments={}",
        args.threshold, args.max_segments
    );

    let models_dir = Path::new("specialist_rf_models");
    let mut router = AcousticHierarchicalRouter::new(args.threshold);

    println!("\nLoading Acoustic Specialist Models:");
    for group in AcousticGroup::all() {
        let filename = format!("specialist_rf_acoustic_{}.json", group.filename_suffix());
        let path = models_dir.join(&filename);
        if path.exists() || path.with_extension("bincode").exists() {
            let _ = router.load(*group, &path);
        }
    }

    println!("\nLoaded {} acoustic specialist models", router.specialists.len());

    // Load segments from cache
    let cache_dir = Path::new("beans_feature_cache_112d");
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let cache_data = fs::read_to_string(cache_dir.join("cache_manifest.json"))?;

    #[derive(Debug, Deserialize)]
    struct M {
        samples: Vec<S>,
    }
    #[derive(Debug, Deserialize)]
    struct S {
        audio_file: String,
        labels: L,
    }
    #[derive(Debug, Deserialize)]
    struct L {
        output: String,
        task: String,
    }

    let manifest: M = serde_json::from_str(&manifest_data)?;
    let cache_manifest: HashMap<String, String> = serde_json::from_str(&cache_data)?;

    let mut segments: Vec<DetectionPayload> = Vec::new();
    let segment_duration_ms = 1000.0;

    for (idx, sample) in manifest.samples.iter().enumerate() {
        if sample.labels.output == "None" || sample.labels.task == "captioning" {
            continue;
        }
        if let Some(cache_file) = cache_manifest.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            let start_time_ms = idx as f64 * segment_duration_ms;
                            segments.push(DetectionPayload {
                                features_112d: features,
                                start_time_ms,
                                end_time_ms: start_time_ms + segment_duration_ms,
                                source_file: sample.audio_file.clone(),
                                segment_idx: idx,
                                true_label: sample.labels.output.clone(),
                            });
                            if args.max_segments > 0 && segments.len() >= args.max_segments {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    println!("\nLoaded {} segments", segments.len());

    println!("\nRunning Detection:");
    let mut detections: Vec<Detection> = Vec::new();
    let mut stats = DetectionStats {
        total_segments: segments.len(),
        positive_detections: 0,
        rejected_by_threshold: 0,
        correct_detections: 0,
        accuracy: 0.0,
        avg_inference_time_us: 0.0,
        acoustic_group_distribution: HashMap::new(),
    };
    let mut total_time_us: f64 = 0.0;

    for (i, seg) in segments.iter().enumerate() {
        if args.verbose && i % 100 == 0 {
            println!("  {}/{}", i, segments.len());
        }

        if let Some((det, time_us)) = router.detect(seg) {
            total_time_us += time_us as f64;
            stats.positive_detections += 1;
            if det.correct {
                stats.correct_detections += 1;
            }
            *stats
                .acoustic_group_distribution
                .entry(det.acoustic_group.clone())
                .or_insert(0) += 1;
            detections.push(det);
        } else {
            stats.rejected_by_threshold += 1;
        }
    }

    stats.accuracy = if stats.positive_detections > 0 {
        stats.correct_detections as f64 / stats.positive_detections as f64
    } else {
        0.0
    };
    stats.avg_inference_time_us = if detections.is_empty() {
        0.0
    } else {
        total_time_us / detections.len() as f64
    };

    println!(
        "\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║  Results                                                              ║");
    println!(
        "╠════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║  Segments:        {:>6}                                              ║",
        stats.total_segments
    );
    println!(
        "║  Detections:      {:>6}  ({:.1}%)                            ║",
        stats.positive_detections,
        stats.positive_detections as f64 / stats.total_segments as f64 * 100.0
    );
    println!(
        "║  Correct:         {:>6}  (acc: {:.1}%)                          ║",
        stats.correct_detections,
        stats.accuracy * 100.0
    );
    println!(
        "║  Rejected:        {:>6}                                              ║",
        stats.rejected_by_threshold
    );
    println!(
        "║  Avg time:        {:>6.1}µs                                      ║",
        stats.avg_inference_time_us
    );
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Acoustic group distribution
    println!("\nAcoustic Group Distribution:");
    let mut groups: Vec<_> = stats.acoustic_group_distribution.iter().collect();
    groups.sort_by(|a, b| b.1.cmp(a.1));
    for (group, count) in &groups {
        println!("  {:<25} {:>6} detections", group, count);
    }

    println!("\nSample Detections:");
    println!("  Start(s)  End(s)    Species                         Conf   Acoustic Group       Correct");
    println!("  ────────  ────────  ──────────────────────────────  ─────  ───────────────────  ───────");
    for d in detections.iter().take(10) {
        println!(
            "  {:>8.3}  {:>8.3}  {:<28} {:>5.1}%  {:<19} {}",
            d.start_s,
            d.end_s,
            d.species,
            d.confidence * 100.0,
            d.acoustic_group,
            if d.correct { "✓" } else { "✗" }
        );
    }

    // Save to JSON
    let output = serde_json::json!({ "detections": detections, "stats": stats });
    fs::write(&args.output, serde_json::to_string_pretty(&output)?)?;
    println!("\nSaved to: {}", args.output);

    Ok(())
}
