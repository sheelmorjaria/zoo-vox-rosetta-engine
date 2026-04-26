//! Train Taxonomic Specialist Random Forests for Hierarchical Ensemble Router
//! ==============================================================================
//!
//! This script trains specialist RF models for each taxonomic group:
//! - RF_Cetacean: Toothed whales (dolphins, porpoises)
//! - RF_Mysticete: Baleen whales (humpback, blue)
//! - RF_Songbird: Passerines (sparrows, finches, warblers)
//! - RF_NonPasserine: Non-passerine birds (parrots, owls)
//! - RF_Amphibian: Frogs and toads
//! - RF_Pinniped: Seals and sea lions
//! - RF_Insect: Crickets, mosquitoes, cicadas
//! - RF_Mammal: Bats, primates, terrestrial mammals
//!
//! Each specialist is trained ONLY on samples from its taxonomic group,
//! allowing it to find subtle splits that would be washed out in a global model.
//!
//! Usage:
//!   cargo run --release --bin train_specialist_rfs
//!
//! Input:
//!   - beans_zero_full_manifest.json: Sample manifest
//!   - beans_feature_cache_112d/: Cached 112D features
//!
//! Output:
//!   - specialist_rf_songbird.json
//!   - specialist_rf_cetacean.json
//!   - ... (one file per specialist)

use anyhow::Result;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

// Import the existing RF implementation from classical_ml
use technical_architecture::classical_ml::RandomForestClassifier;

// =============================================================================
// Feature Dimensions
// =============================================================================

const FEATURE_DIM: usize = 112;

// =============================================================================
// Hyperparameters for Specialist RFs
// =============================================================================

/// Number of trees per specialist
const N_ESTIMATORS: usize = 200;

/// Maximum depth per tree
const MAX_DEPTH: usize = 30;

/// Minimum samples to split
const MIN_SAMPLES_SPLIT: usize = 5;

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    #[allow(dead_code)]
    dataset: String,
    #[allow(dead_code)]
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansSample {
    audio_file: String,
    #[allow(dead_code)]
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: String,
    task: String,
}

#[derive(Debug, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
    #[allow(dead_code)]
    feature_count: usize,
}

/// Raw dataset for 112D features (bypasses 45D assertion)
struct RawDataset {
    features: Array2<f32>,
    labels: Vec<String>,
    label_to_idx: HashMap<String, usize>,
    idx_to_label: HashMap<usize, String>,
}

impl RawDataset {
    fn len(&self) -> usize {
        self.labels.len()
    }

    fn num_classes(&self) -> usize {
        self.label_to_idx.len()
    }

    /// Split into train/test sets
    fn train_test_split(&self, test_ratio: f32, seed: u64) -> (Self, Self) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let n_samples = self.labels.len();
        let n_test = (n_samples as f32 * test_ratio) as usize;
        let n_train = n_samples - n_test;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let train_indices: Vec<usize> = indices[..n_train].to_vec();
        let test_indices: Vec<usize> = indices[n_train..].to_vec();

        // Create train dataset
        let mut train_features = Array2::zeros((n_train, FEATURE_DIM));
        let train_labels: Vec<String> = train_indices.iter().map(|&i| self.labels[i].clone()).collect();

        for (j, &i) in train_indices.iter().enumerate() {
            for k in 0..FEATURE_DIM {
                train_features[[j, k]] = self.features[[i, k]];
            }
        }

        // Create test dataset
        let mut test_features = Array2::zeros((n_test, FEATURE_DIM));
        let test_labels: Vec<String> = test_indices.iter().map(|&i| self.labels[i].clone()).collect();

        for (j, &i) in test_indices.iter().enumerate() {
            for k in 0..FEATURE_DIM {
                test_features[[j, k]] = self.features[[i, k]];
            }
        }

        (
            RawDataset {
                features: train_features,
                labels: train_labels,
                label_to_idx: self.label_to_idx.clone(),
                idx_to_label: self.idx_to_label.clone(),
            },
            RawDataset {
                features: test_features,
                labels: test_labels,
                label_to_idx: self.label_to_idx.clone(),
                idx_to_label: self.idx_to_label.clone(),
            },
        )
    }
}

// =============================================================================
// Taxonomic Mapping
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Taxon {
    Cetacean,
    Mysticete,
    Songbird,
    NonPasserine,
    Amphibian,
    Pinniped,
    Insect,
    Mammal,
    Unknown,
}

fn map_species_to_taxon(species: &str) -> Taxon {
    let species_lower = species.to_lowercase();

    // Cetaceans (toothed whales)
    if species_lower.contains("dolphin")
        || species_lower.contains("porpoise")
        || species_lower.contains("orca")
        || species_lower.contains("sperm whale")
        || species_lower.contains("beaked")
        || species_lower.contains("delphinid")
        || species_lower.contains("phocoen")
    {
        return Taxon::Cetacean;
    }

    // Mysticetes (baleen whales)
    if species_lower.contains("humpback")
        || species_lower.contains("blue whale")
        || species_lower.contains("fin whale")
        || species_lower.contains("minke")
        || species_lower.contains("gray whale")
        || species_lower.contains("right whale")
        || species_lower.contains("bowhead")
        || species_lower.contains("balaenopter")
    {
        return Taxon::Mysticete;
    }

    // Pinnipeds
    if species_lower.contains("seal")
        || species_lower.contains("sea lion")
        || species_lower.contains("walrus")
        || species_lower.contains("phocid")
        || species_lower.contains("otariid")
    {
        return Taxon::Pinniped;
    }

    // Songbirds (passerines)
    if species_lower.contains("sparrow")
        || species_lower.contains("finch")
        || species_lower.contains("warbler")
        || species_lower.contains("thrush")
        || species_lower.contains("robin")
        || species_lower.contains("cardinal")
        || species_lower.contains("towhee")
        || species_lower.contains("ovenbird")
        || species_lower.contains("wren")
        || species_lower.contains("tit")
        || species_lower.contains("swainson")
    {
        return Taxon::Songbird;
    }

    // Non-passerine birds
    if species_lower.contains("parrot")
        || species_lower.contains("owl")
        || species_lower.contains("hawk")
        || species_lower.contains("eagle")
        || species_lower.contains("duck")
        || species_lower.contains("goose")
        || species_lower.contains("gull")
        || species_lower.contains("crow")
        || species_lower.contains("raven")
        || species_lower.contains("penguin")
        || species_lower.contains("psittacid")
        || species_lower.contains("strigid")
    {
        return Taxon::NonPasserine;
    }

    // Anurans (frogs/toads)
    if species_lower.contains("frog")
        || species_lower.contains("toad")
        || species_lower.contains("ranid")
        || species_lower.contains("bufonid")
        || species_lower.contains("hylid")
        || species_lower.contains("peeper")
    {
        return Taxon::Amphibian;
    }

    // Insects
    if species_lower.contains("cricket")
        || species_lower.contains("mosquito")
        || species_lower.contains("cicada")
        || species_lower.contains("grasshopper")
        || species_lower.contains("katydid")
        || species_lower.contains("bee")
        || species_lower.contains("fly")
        || species_lower.contains("anopheles")
        || species_lower.contains("aedes")
        || species_lower.contains("culex")
        || species_lower.contains("culicid")
    {
        return Taxon::Insect;
    }

    // Bats (mammals with FM)
    if species_lower.contains("bat")
        || species_lower.contains("pteropodid")
        || species_lower.contains("vesper")
        || species_lower.contains("phyllostomid")
    {
        return Taxon::Mammal;
    }

    // Primates and other mammals
    if species_lower.contains("monkey")
        || species_lower.contains("ape")
        || species_lower.contains("gibbon")
        || species_lower.contains("chimp")
        || species_lower.contains("gorilla")
        || species_lower.contains("primate")
    {
        return Taxon::Mammal;
    }

    // Default to Unknown
    Taxon::Unknown
}

fn taxon_name(taxon: Taxon) -> &'static str {
    match taxon {
        Taxon::Cetacean => "cetacean",
        Taxon::Mysticete => "mysticete",
        Taxon::Songbird => "songbird",
        Taxon::NonPasserine => "non_passerine",
        Taxon::Amphibian => "amphibian",
        Taxon::Pinniped => "pinniped",
        Taxon::Insect => "insect",
        Taxon::Mammal => "mammal",
        Taxon::Unknown => "unknown",
    }
}

// =============================================================================
// Data Loading
// =============================================================================

fn load_data() -> Result<HashMap<Taxon, RawDataset>> {
    let manifest_path = "beans_zero_full_manifest.json";
    println!("Loading manifest from: {}", manifest_path);
    let manifest_data = fs::read_to_string(manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples: {}", manifest.samples.len());

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest_path = cache_dir.join("cache_manifest.json");
    println!("Loading cache manifest from: {:?}", cache_manifest_path);
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Group samples by taxon
    let mut taxon_samples: HashMap<Taxon, Vec<(Vec<f32>, String)>> = HashMap::new();

    for sample in &manifest.samples {
        let audio_file = &sample.audio_file;
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        let taxon = map_species_to_taxon(&label);

        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            taxon_samples.entry(taxon).or_default().push((features, label));
                        }
                    }
                }
            }
        }
    }

    // Convert to RawDataset
    let mut result: HashMap<Taxon, RawDataset> = HashMap::new();

    for (taxon, samples) in taxon_samples {
        if samples.is_empty() {
            continue;
        }

        // Build label mapping for this taxon
        let mut unique_labels: Vec<String> = samples.iter().map(|(_, l)| l.clone()).collect();
        unique_labels.sort();
        unique_labels.dedup();

        let mut label_to_idx: HashMap<String, usize> = HashMap::new();
        let mut idx_to_label: HashMap<usize, String> = HashMap::new();
        for (idx, label) in unique_labels.iter().enumerate() {
            label_to_idx.insert(label.clone(), idx);
            idx_to_label.insert(idx, label.clone());
        }

        // Build feature matrix
        let n_samples = samples.len();
        let mut features_array = Array2::<f32>::zeros((n_samples, FEATURE_DIM));
        let labels: Vec<String> = samples.iter().map(|(_, l)| l.clone()).collect();

        for (i, (feat, _)) in samples.iter().enumerate() {
            for (j, &val) in feat.iter().enumerate() {
                features_array[[i, j]] = val;
            }
        }

        println!(
            "  {}: {} samples, {} classes",
            taxon_name(taxon),
            n_samples,
            unique_labels.len()
        );

        result.insert(
            taxon,
            RawDataset {
                features: features_array,
                labels,
                label_to_idx,
                idx_to_label,
            },
        );
    }

    Ok(result)
}

// =============================================================================
// Training
// =============================================================================

/// Train a specialist RF for a taxonomic group
fn train_specialist(taxon: Taxon, dataset: &RawDataset) -> Result<RandomForestClassifier> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Specialist: {:<43}║", format!("{:?}", taxon));
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Samples: {:>6}  Classes: {:>5}                              ║",
        dataset.len(),
        dataset.num_classes()
    );
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    let start = Instant::now();

    // Split into train/test (90/10)
    let (train, test) = dataset.train_test_split(0.1, 42);

    println!("  Train: {} samples, Test: {} samples", train.len(), test.len());

    // Create and train RF with balanced class weights
    let mut rf = RandomForestClassifier::new(N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT).with_balanced_weights();

    // Use fit_raw to bypass 45D assertion
    rf.fit_raw(&train.features, &train.labels, &train.label_to_idx, &train.idx_to_label)?;

    let train_time = start.elapsed();
    println!("  Training time: {:.2}s", train_time.as_secs_f32());

    // Evaluate on test set
    let preds = rf.predict_batch(&test.features);
    let correct = preds
        .iter()
        .zip(test.labels.iter())
        .filter(|(pred, label)| {
            let pred_idx = test.label_to_idx.get(*label).copied().unwrap_or(0);
            **pred == pred_idx
        })
        .count();

    let accuracy = if test.len() > 0 {
        correct as f32 / test.len() as f32 * 100.0
    } else {
        0.0
    };

    println!("  Test accuracy: {:.2}% ({}/{})", accuracy, correct, test.len());

    // Show top feature importances
    let importances = rf.feature_importances();
    let mut indexed: Vec<(usize, f32)> = importances.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top 5 features:");
    for (i, (idx, imp)) in indexed.iter().take(5).enumerate() {
        println!("    {}: Feature {} = {:.4}", i + 1, idx, imp);
    }

    Ok(rf)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     Taxonomic Specialist RF Training (Parallel Rust)              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  N_estimators: {}", N_ESTIMATORS);
    println!("  Max_depth: {}", MAX_DEPTH);
    println!("  Min_samples_split: {}", MIN_SAMPLES_SPLIT);
    println!("  Parallel: Specialists + Tree building (rayon)");
    println!();

    // Load data grouped by taxon
    let taxon_data = load_data()?;

    println!();
    println!("Loaded data for {} taxonomic groups", taxon_data.len());
    println!();

    // Create output directory
    let output_dir = Path::new("specialist_rf_models");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    // Define taxons to train
    let taxons_to_train: Vec<Taxon> = vec![
        Taxon::Songbird,
        Taxon::Cetacean,
        Taxon::Mysticete,
        Taxon::NonPasserine,
        Taxon::Amphibian,
        Taxon::Pinniped,
        Taxon::Insect,
        Taxon::Mammal,
    ];

    // Train specialists in parallel
    let output_dir = Arc::new(output_dir.to_path_buf());
    let results: Vec<(Taxon, Result<String>)> = taxons_to_train
        .into_par_iter()
        .filter_map(|taxon| {
            let dataset = taxon_data.get(&taxon)?;

            // Skip if too few samples
            if dataset.len() < 10 {
                println!("\nSkipping {:?}: only {} samples", taxon, dataset.len());
                return None;
            }

            let output_dir = Arc::clone(&output_dir);

            // Train specialist
            let result = train_specialist(taxon, dataset).and_then(|rf| {
                // Save to bincode (much smaller and faster to load)
                let model_path = output_dir.join(format!("specialist_rf_{}.bincode", taxon_name(taxon)));
                let file = fs::File::create(&model_path)?;
                let writer = BufWriter::new(file);
                bincode::serialize_into(writer, &rf)?;
                Ok(format!("{:?}", model_path))
            });

            Some((taxon, result))
        })
        .collect();

    // Report results
    let mut trained_count = 0;
    for (taxon, result) in results {
        match result {
            Ok(path) => {
                println!("  Saved: {}", path);
                trained_count += 1;
            }
            Err(e) => {
                println!("\nError training {:?}: {}", taxon, e);
            }
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Complete                                                 ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Specialists trained: {:>3}                                          ║",
        trained_count
    );
    println!("║  Models saved to: specialist_rf_models/                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
