//! Bioacoustic Detection Pipeline with Timestamps
//!
//! Simple pipeline for detection with timestamps

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::classical_ml::RandomForestClassifier;
use technical_architecture::taxonomic_router::{map_species_to_taxon, Taxon, FEATURE_DIM};

#[derive(Parser, Debug)]
#[command(name = "detection_pipeline", version = "1.0")]
struct Args {
    #[arg(short, long, default_value = "detections.json")]
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
    pub taxon: String,
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
    pub taxon_distribution: HashMap<String, usize>,
}

fn build_label_canonical_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mappings = [
        ("Dall's Porpoise", "Phocoenoides dalli"),
        ("Harbor Porpoise", "Phocoena phocoena"),
        ("Finless Porpoise", "Neophocaena phocaenoides"),
        ("Bottlenose Dolphin", "Tursiops truncatus"),
        ("Common Dolphin", "Delphinus delphis"),
        ("Killer Whale", "Orcinus orca"),
        ("Humpback Whale", "Megaptera novaeangliae"),
        ("Minke Whale", "Balaenoptera acutorostrata"),
        ("Minke whale", "Balaenoptera acutorostrata"),
    ];
    for (common, scientific) in &mappings {
        map.insert(common.to_lowercase(), scientific.to_string());
        map.insert(scientific.to_lowercase(), scientific.to_string());
    }
    map
}

fn normalize_label(label: &str, canonical_map: &HashMap<String, String>) -> String {
    canonical_map
        .get(&label.to_lowercase())
        .cloned()
        .unwrap_or_else(|| label.to_string())
}

struct HierarchicalRouter {
    specialists: HashMap<Taxon, RandomForestClassifier>,
    canonical_map: HashMap<String, String>,
    threshold: f32,
}

impl HierarchicalRouter {
    fn new(threshold: f32) -> Self {
        Self {
            specialists: HashMap::new(),
            canonical_map: build_label_canonical_map(),
            threshold,
        }
    }

    fn load(&mut self, taxon: Taxon, path: &Path) -> Result<()> {
        let start = Instant::now();
        let data = fs::read_to_string(path)?;
        let model: RandomForestClassifier = serde_json::from_str(&data)?;
        println!(
            "  {:?}: {} trees, {} classes ({:.2}s)",
            taxon,
            model.n_trees(),
            model.n_classes(),
            start.elapsed().as_secs_f32()
        );
        self.specialists.insert(taxon, model);
        Ok(())
    }

    fn detect(&self, payload: &DetectionPayload) -> Option<(Detection, u64)> {
        let taxon = map_species_to_taxon(&payload.true_label);
        let model = self.specialists.get(&taxon)?;
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
                taxon: format!("{:?}", taxon),
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

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Bioacoustic Detection Pipeline                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!(
        "\nConfig: threshold={:.2}, max_segments={}",
        args.threshold, args.max_segments
    );

    let models_dir = Path::new("specialist_rf_models");
    let mut router = HierarchicalRouter::new(args.threshold);

    println!("\nLoading Models:");
    for (taxon, filename) in [
        (Taxon::Songbird, "specialist_rf_songbird.json"),
        (Taxon::Cetacean, "specialist_rf_cetacean.json"),
        (Taxon::Mysticete, "specialist_rf_mysticete.json"),
        (Taxon::NonPasserine, "specialist_rf_non_passerine.json"),
        (Taxon::Amphibian, "specialist_rf_amphibian.json"),
        (Taxon::Pinniped, "specialist_rf_pinniped.json"),
        (Taxon::Insect, "specialist_rf_insect.json"),
        (Taxon::Mammal, "specialist_rf_mammal.json"),
    ] {
        let path = models_dir.join(filename);
        if path.exists() {
            let _ = router.load(taxon, &path);
        }
    }

    println!("\nLoaded {} specialist models", router.specialists.len());

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
    let _sample_rate = 48000.0;
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
        taxon_distribution: HashMap::new(),
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
            *stats.taxon_distribution.entry(det.taxon.clone()).or_insert(0) += 1;
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
    println!("║  Results                                                          ║");
    println!(
        "╠═════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║  Segments:        {:>6}                                        ║",
        stats.total_segments
    );
    println!(
        "║  Detections:      {:>6}  ({:.1}%)                              ║",
        stats.positive_detections,
        stats.positive_detections as f64 / stats.total_segments as f64 * 100.0
    );
    println!(
        "║  Correct:         {:>6}  (acc: {:.1}%)                          ║",
        stats.correct_detections,
        stats.accuracy * 100.0
    );
    println!(
        "║  Rejected:        {:>6}                                        ║",
        stats.rejected_by_threshold
    );
    println!(
        "║  Avg time:        {:>6.1}µs                                ║",
        stats.avg_inference_time_us
    );
    println!("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    println!("\nSample Detections:");
    println!("  Start(s)  End(s)   Species                         Conf   Correct");
    println!("  ─────────  ─────────  ────────────────────────────────  ───────");
    for d in detections.iter().take(10) {
        println!(
            "  {:>8.3}    {:>8.3}    {:<30} {:>5.1}%   {}",
            d.start_s,
            d.end_s,
            d.species,
            d.confidence * 100.0,
            if d.correct { "✓" } else { "✗" }
        );
    }

    // Save to JSON
    let output = serde_json::json!({ "detections": detections, "stats": stats });
    fs::write(&args.output, serde_json::to_string_pretty(&output)?)?;
    println!("\nSaved to: {}", args.output);

    Ok(())
}
