//! Zero-Shot Classification Evaluation
//! ====================================
//!
//! Evaluates the zero-shot classification system on BEANS-Zero benchmark.
//!
//! Usage:
//!   cargo run --release --bin eval_zero_shot -- <manifest.json> [--gallery gallery.json]
//!
//! The manifest should contain samples with features and labels.
//! The gallery contains reference embeddings for known species.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use technical_architecture::zero_shot_router::{
    ReferenceGallery, ReferenceSample, SiameseEmbedding, ZeroShotConfig, ZeroShotRouter, FEATURE_DIM,
};

// ============================================================================
// Manifest Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct Manifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<ManifestSample>,
}

#[derive(Debug, Deserialize)]
struct ManifestSample {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    audio_file: Option<String>,
    #[serde(default)]
    features: Option<Vec<f32>>,
    #[serde(default)]
    n_samples: Option<u32>,
    #[serde(default)]
    labels: ManifestLabels,
}

impl ManifestSample {
    fn get_id(&self) -> String {
        self.id
            .clone()
            .or(self.audio_file.clone())
            .unwrap_or_else(|| format!("sample_{}", self.n_samples.unwrap_or(0)))
    }
}

#[derive(Debug, Deserialize, Default)]
struct ManifestLabels {
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    species: Option<String>,
    #[serde(default)]
    #[allow(dead_code)] // Field exists for JSON deserialization compatibility
    task: Option<String>,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: eval_zero_shot <manifest.json> [--gallery gallery.json]");
        std::process::exit(1);
    }

    let manifest_path = &args[1];
    let gallery_path = args
        .get(2)
        .and_then(|s| s.strip_prefix("--gallery="))
        .unwrap_or("reference_gallery.json");

    println!("{}", "=".repeat(80));
    println!("ZERO-SHOT CLASSIFICATION EVALUATION");
    println!("{}", "=".repeat(80));

    // Load manifest
    println!("\n[1] Loading manifest from: {}", manifest_path);
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(manifest_path).with_context(|| "Failed to read manifest")?)?;
    println!("   Dataset: {}", manifest.dataset);
    println!("   Samples: {}", manifest.n_samples);

    // Load or create gallery
    let mut gallery = if PathBuf::from(gallery_path).exists() {
        println!("\n[2] Loading gallery from: {}", gallery_path);
        ReferenceGallery::load_from_json(gallery_path).map_err(|e| anyhow::anyhow!("Failed to load gallery: {}", e))?
    } else {
        println!("\n[2] Building gallery from manifest features...");
        build_gallery_from_manifest(&manifest)?
    };
    println!("   Gallery size: {} samples", gallery.len());

    // Configure zero-shot router
    // Testing with raw features and euclidean distance for better discrimination
    let config = ZeroShotConfig {
        k_neighbors: 1,           // 1-NN for small gallery
        distance_threshold: 1.0,  // Allow all matches
        min_confidence: 0.0,      // No confidence threshold
        apply_reweighting: false, // Disable reweighting
        weighted_knn: false,
        distance_metric: "euclidean".to_string(), // Use euclidean on raw features
    };

    println!("\n[3] Configuration:");
    println!("   k-neighbors: {}", config.k_neighbors);
    println!("   Distance threshold: {:.3}", config.distance_threshold);
    println!("   Min confidence: {:.1}%", config.min_confidence * 100.0);

    // Create embedding model and regenerate gallery embeddings for consistency
    // This ensures query and gallery embeddings use the same weights
    let embedding = SiameseEmbedding::default();
    gallery.regenerate_embeddings(&embedding);
    println!("   Regenerated gallery embeddings for consistency");

    // Create router with custom embedding
    let router = ZeroShotRouter::with_embedding(config, gallery, embedding)
        .map_err(|e| anyhow::anyhow!("Failed to create zero-shot router: {}", e))?;

    // Evaluate
    println!("\n[4] Running zero-shot classification...");
    let start_time = Instant::now();

    let mut results_by_type = HashMap::new();
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut species_correct = HashMap::new();
    let mut species_total = HashMap::new();

    for sample in &manifest.samples {
        // Get features
        let features = match &sample.features {
            Some(f) => f.clone(),
            None => {
                // Generate synthetic features if not available
                vec![0.0; FEATURE_DIM]
            }
        };

        if features.len() != FEATURE_DIM {
            continue;
        }

        // Get true label
        let true_species = sample
            .labels
            .species
            .as_ref()
            .or(sample.labels.output.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        // Classify
        match router.classify(&features) {
            Ok(result) => {
                total += 1;

                // Track by prediction type
                let type_key = format!("{:?}", result.prediction_type);
                *results_by_type.entry(type_key).or_insert(0) += 1;

                // Track accuracy
                if result.species == true_species {
                    correct += 1;
                }

                // Track per-species accuracy
                *species_total.entry(true_species.to_string()).or_insert(0) += 1;
                if result.species == true_species {
                    *species_correct.entry(true_species.to_string()).or_insert(0) += 1;
                }
            }
            Err(e) => {
                eprintln!("Error classifying sample {}: {}", sample.get_id(), e);
            }
        }
    }

    let elapsed = start_time.elapsed();

    // Print results
    println!("\n{}", "=".repeat(80));
    println!("ZERO-SHOT EVALUATION RESULTS");
    println!("{}", "=".repeat(80));

    println!("\n[OVERALL METRICS]");
    println!("   Total samples: {}", total);
    println!("   Processing time: {:?}", elapsed);
    println!("   Samples/second: {:.1}", total as f64 / elapsed.as_secs_f64());

    if total > 0 {
        let accuracy = correct as f64 / total as f64 * 100.0;
        println!("\n[ACCURACY]");
        println!("   Overall: {}/{} = {:.2}%", correct, total, accuracy);

        println!("\n[PREDICTION TYPE DISTRIBUTION]");
        for (ptype, count) in &results_by_type {
            let pct = *count as f64 / total as f64 * 100.0;
            println!("   {}: {} ({:.1}%)", ptype, count, pct);
        }

        // Per-species breakdown (top 10)
        let mut species_acc: Vec<_> = species_total
            .iter()
            .map(|(s, &t)| {
                let c = species_correct.get(s).copied().unwrap_or(0);
                (s.clone(), c, t)
            })
            .collect();
        species_acc.sort_by(|a, b| b.2.cmp(&a.2));

        println!("\n[PER-SPECIES ACCURACY (Top 10)]");
        for (species, c, t) in species_acc.iter().take(10) {
            let acc = if *t > 0 { *c as f64 / *t as f64 * 100.0 } else { 0.0 };
            println!("   {}: {}/{} = {:.1}%", species, c, t, acc);
        }
    }

    // Save results
    let output = serde_json::json!({
        "total_samples": total,
        "correct": correct,
        "accuracy": if total > 0 { correct as f64 / total as f64 } else { 0.0 },
        "processing_time_ms": elapsed.as_millis(),
        "prediction_types": results_by_type,
    });

    std::fs::write("zero_shot_results.json", serde_json::to_string_pretty(&output)?)?;
    println!("\nResults saved to: zero_shot_results.json");

    println!("\n{}", "=".repeat(80));
    println!("Zero-Shot Evaluation Complete.");
    println!("{}", "=".repeat(80));

    Ok(())
}

fn build_gallery_from_manifest(manifest: &Manifest) -> Result<ReferenceGallery> {
    let mut gallery = ReferenceGallery::new();
    let embedding = SiameseEmbedding::default();

    let mut seen_species: HashMap<String, usize> = HashMap::new();

    for sample in &manifest.samples {
        // Get species label
        let species = sample
            .labels
            .species
            .as_ref()
            .or(sample.labels.output.as_ref())
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string());

        // Skip if we already have this species
        if seen_species.contains_key(&species) {
            continue;
        }

        // Get features
        let features = match &sample.features {
            Some(f) if f.len() == FEATURE_DIM => f.clone(),
            _ => continue,
        };

        // Generate embedding
        let latent = embedding.embed(&features);

        // Determine taxon from species name (heuristic)
        let taxon = if species.contains("bat") || species.contains("Bat") {
            technical_architecture::taxonomic_router::Taxon::Mammal
        } else if species.contains("bird") || species.contains("Bird") || species.contains("finch") {
            technical_architecture::taxonomic_router::Taxon::Songbird
        } else if species.contains("dolphin") || species.contains("whale") {
            technical_architecture::taxonomic_router::Taxon::Cetacean
        } else if species.contains("frog") {
            technical_architecture::taxonomic_router::Taxon::Amphibian
        } else {
            technical_architecture::taxonomic_router::Taxon::Unknown
        };

        gallery.add_sample(ReferenceSample {
            species: species.clone(),
            taxon,
            embedding: latent,
            original_features: Some(features),
        });

        seen_species.insert(species, 1);
    }

    Ok(gallery)
}
