//! Train Acoustic Specialist Random Forests for Hierarchical Ensemble Router
//! ==============================================================================
//!
//! This script trains specialist RF models using ACOUSTIC COHERENCE instead of
//! biological taxonomy. The key insight is that broad biological groups are
//! often acoustically incoherent.
//!
//! Acoustic Splits:
//!
//! 1. MAMMALS (3-way split):
//!    - UltrasonicMammal: Bats (F0 20-80kHz, duration 5-50ms)
//!    - SonicLongMammal: Whales (F0 20-5000Hz, duration 500-5000ms)
//!    - SonicShortMammal: Primates, other land mammals
//!
//! 2. INSECTS (2-way split):
//!    - InsectWingbeat: Mosquitoes, flies, bees (steady F0, pure tones)
//!    - InsectStridulation: Crickets, cicadas (broadband, impulsive)
//!
//! 3. BIRDS (3-way split):
//!    - BirdHighFreq: Songbirds, wrens, warblers (high F0, fast modulation)
//!    - BirdLowFreq: Doves, pigeons, cuckoos (low F0, long duration)
//!    - BirdMechanical: Hummingbirds (broadband, pulse-like)
//!
//! 4. MARINE MAMMALS (3-way split):
//!    - MarineWhistle: Dolphins, orcas (FM sweeps, harmonic)
//!    - MarineClick: Sperm whales, porpoises (impulsive, broadband)
//!    - MarineMoan: Humpbacks, blue whales (low F0, long duration)
//!
//! Expected: 37% -> 80-90% accuracy for split groups
//!
//! Usage:
//!   cargo run --release --bin train_acoustic_specialist_rfs

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

use technical_architecture::classical_ml::RandomForestClassifier;

// =============================================================================
// Constants
// =============================================================================

const FEATURE_DIM: usize = 112;
const N_ESTIMATORS: usize = 200;
const MAX_DEPTH: usize = 30;
const MIN_SAMPLES_SPLIT: usize = 5;

// =============================================================================
// Acoustic Groups
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcousticGroup {
    // Mammals (3-way split)
    UltrasonicMammal, // Bats: 20-80kHz, 5-50ms
    SonicLongMammal,  // Whales: 20-5000Hz, 500-5000ms
    SonicShortMammal, // Primates: mid F0, variable

    // Insects (2-way split)
    InsectWingbeat,     // Mosquitoes, flies: steady F0, pure tones
    InsectStridulation, // Crickets, cicadas: broadband, impulsive

    // Birds (3-way split)
    BirdHighFreq,   // Songbirds: high F0, fast modulation
    BirdLowFreq,    // Doves, owls: low F0, long duration
    BirdMechanical, // Hummingbirds: broadband, pulse-like

    // Marine Mammals (3-way split)
    MarineWhistle, // Dolphins: FM sweeps, harmonic
    MarineClick,   // Porpoises: impulsive, broadband
    MarineMoan,    // Baleen whales: low F0, long duration

    // Other
    Amphibian, // Frogs, toads
    Pinniped,  // Seals, sea lions
}

fn map_species_to_acoustic_group(species: &str) -> AcousticGroup {
    let s = species.to_lowercase();

    // === ULTRASONIC MAMMALS (Bats) ===
    if s.contains("bat")
        || s.contains("pteropodid")
        || s.contains("vesper")
        || s.contains("phyllostomid")
        || s.contains("rhinolophus")
        || s.contains("myotis")
        || s.contains("nyctalus")
        || s.contains("pipistrellus")
        || s.contains("eptesicus")
        || s.contains("plecotus")
        || s.contains("miniopterus")
        || s.contains("tadarida")
        || s.contains("molossid")
        || s.contains("vespertilion")
        || s.contains("noctilio")
        || s.contains("hypsignathus")
        || s.contains("pteropus")
        || s.contains("nyctinomops")
        || s.contains("molossus")
    {
        return AcousticGroup::UltrasonicMammal;
    }

    // === SONIC LONG MAMMALS (Baleen Whales) ===
    if s.contains("humpback")
        || s.contains("blue whale")
        || s.contains("fin whale")
        || s.contains("minke")
        || s.contains("gray whale")
        || s.contains("grey whale")
        || s.contains("right whale")
        || s.contains("bowhead")
        || s.contains("balaenopter")
        || s.contains("megaptera")
        || s.contains("eschrichtius")
        || s.contains("balaena")
    {
        return AcousticGroup::SonicLongMammal;
    }

    // === MARINE WHISTLE (Dolphins, Orcas) ===
    if s.contains("dolphin")
        || s.contains("delphin")
        || s.contains("orca")
        || s.contains("killer whale")
        || s.contains("pilot whale")
        || s.contains("tursiops")
        || s.contains("grampus")
        || s.contains("stenella")
        || s.contains("lagenorhynchus")
        || s.contains("delphinapterus")
    {
        return AcousticGroup::MarineWhistle;
    }

    // === MARINE CLICK (Porpoises, Sperm Whales) ===
    if s.contains("porpoise")
        || s.contains("phocoen")
        || s.contains("sperm whale")
        || s.contains("physeter")
        || s.contains("beaked whale")
        || s.contains("ziphius")
        || s.contains("mesoplodon")
        || s.contains("kogia")
    {
        return AcousticGroup::MarineClick;
    }

    // === MARINE MOAN (Baleen whales - fallback) ===
    if s.contains("whale") && !s.contains("killer") {
        return AcousticGroup::MarineMoan;
    }

    // === PINNIPEDS ===
    if s.contains("seal")
        || s.contains("sea lion")
        || s.contains("walrus")
        || s.contains("phocid")
        || s.contains("otariid")
        || s.contains("otary")
    {
        return AcousticGroup::Pinniped;
    }

    // === INSECT WINGBEAT (Pure tones) ===
    if s.contains("mosquito")
        || s.contains("aedes")
        || s.contains("anopheles")
        || s.contains("culex")
        || s.contains("culicid")
        || s.contains("fly")
        || s.contains("muscidae")
        || s.contains("bee")
        || s.contains("apis")
        || s.contains("bombus")
        || s.contains("wasp")
        || s.contains("syrphid")
    {
        return AcousticGroup::InsectWingbeat;
    }

    // === INSECT STRIDULATION (Broadband pulses) ===
    if s.contains("cricket")
        || s.contains("cicada")
        || s.contains("grasshopper")
        || s.contains("katydid")
        || s.contains("tettigoniid")
        || s.contains("gryllid")
        || s.contains("acridid")
        || s.contains("orthoptera")
    {
        return AcousticGroup::InsectStridulation;
    }

    // === BIRD HIGH FREQ (Songbirds) ===
    if s.contains("sparrow")
        || s.contains("finch")
        || s.contains("warbler")
        || s.contains("thrush")
        || s.contains("robin")
        || s.contains("cardinal")
        || s.contains("towhee")
        || s.contains("ovenbird")
        || s.contains("wren")
        || s.contains("tit")
        || s.contains("swainson")
        || s.contains("junco")
        || s.contains("bunting")
        || s.contains("blackbird")
        || s.contains("meadowlark")
        || s.contains("cowbird")
        || s.contains("oriole")
        || s.contains("grackle")
        || s.contains("bobolink")
        || s.contains("lark")
        || s.contains("pipit")
        || s.contains("longspur")
        || s.contains("bluebird")
        || s.contains("solitaire")
        || s.contains("passerine")
        || s.contains("passer")
    {
        return AcousticGroup::BirdHighFreq;
    }

    // === BIRD LOW FREQ (Non-passerines with low calls) ===
    if s.contains("dove")
        || s.contains("pigeon")
        || s.contains("owl")
        || s.contains("cuckoo")
        || s.contains("quail")
        || s.contains("grouse")
        || s.contains("turkey")
        || s.contains("goose")
        || s.contains("swan")
        || s.contains("heron")
        || s.contains("stork")
        || s.contains("crane")
        || s.contains("columb")
        || s.contains("strigid")
    {
        return AcousticGroup::BirdLowFreq;
    }

    // === BIRD MECHANICAL (Hummingbirds, snipe) ===
    if s.contains("hummingbird")
        || s.contains("trochilid")
        || s.contains("snipe")
        || s.contains("gallinago")
        || s.contains("woodpecker")
        || s.contains("picid")
    {
        return AcousticGroup::BirdMechanical;
    }

    // === OTHER NON-PASSERINE BIRDS ===
    if s.contains("parrot")
        || s.contains("hawk")
        || s.contains("eagle")
        || s.contains("duck")
        || s.contains("gull")
        || s.contains("crow")
        || s.contains("raven")
        || s.contains("penguin")
        || s.contains("psittacid")
        || s.contains("bird")
    {
        return AcousticGroup::BirdLowFreq;
    }

    // === AMPHIBIANS ===
    if s.contains("frog")
        || s.contains("toad")
        || s.contains("ranid")
        || s.contains("bufonid")
        || s.contains("hylid")
        || s.contains("peeper")
        || s.contains("anuran")
    {
        return AcousticGroup::Amphibian;
    }

    // === SONIC SHORT MAMMALS (Primates, land mammals) ===
    if s.contains("monkey")
        || s.contains("ape")
        || s.contains("gibbon")
        || s.contains("chimp")
        || s.contains("gorilla")
        || s.contains("primate")
        || s.contains("marmoset")
        || s.contains("lemur")
        || s.contains("tamarin")
        || s.contains("capuchin")
        || s.contains("macaque")
        || s.contains("howler")
    {
        return AcousticGroup::SonicShortMammal;
    }

    // Default
    AcousticGroup::SonicShortMammal
}

fn acoustic_group_name(group: AcousticGroup) -> &'static str {
    match group {
        AcousticGroup::UltrasonicMammal => "ultrasonic_mammal",
        AcousticGroup::SonicLongMammal => "sonic_long_mammal",
        AcousticGroup::SonicShortMammal => "sonic_short_mammal",
        AcousticGroup::InsectWingbeat => "insect_wingbeat",
        AcousticGroup::InsectStridulation => "insect_stridulation",
        AcousticGroup::BirdHighFreq => "bird_high_freq",
        AcousticGroup::BirdLowFreq => "bird_low_freq",
        AcousticGroup::BirdMechanical => "bird_mechanical",
        AcousticGroup::MarineWhistle => "marine_whistle",
        AcousticGroup::MarineClick => "marine_click",
        AcousticGroup::MarineMoan => "marine_moan",
        AcousticGroup::Amphibian => "amphibian",
        AcousticGroup::Pinniped => "pinniped",
    }
}

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansSample {
    audio_file: String,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: String,
    #[allow(dead_code)] // Field exists for JSON deserialization compatibility
    task: String,
}

#[derive(Debug, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
}

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

    fn train_test_split(&self, test_ratio: f32, seed: u64) -> (Self, Self) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let n_samples = self.labels.len();
        let n_test = (n_samples as f32 * test_ratio) as usize;
        let n_train = n_samples - n_test;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let train_indices: Vec<usize> = indices[..n_train].to_vec();
        let test_indices: Vec<usize> = indices[n_train..].to_vec();

        let mut train_features = Array2::zeros((n_train, FEATURE_DIM));
        let train_labels: Vec<String> = train_indices.iter().map(|&i| self.labels[i].clone()).collect();

        for (j, &i) in train_indices.iter().enumerate() {
            for k in 0..FEATURE_DIM {
                train_features[[j, k]] = self.features[[i, k]];
            }
        }

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
// Data Loading
// =============================================================================

fn load_data() -> Result<HashMap<AcousticGroup, RawDataset>> {
    println!("Loading manifest...");
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples: {}", manifest.samples.len());

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_data = fs::read_to_string(cache_dir.join("cache_manifest.json"))?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features: {}", cache_manifest.entries.len());

    // Group samples by acoustic group
    let mut group_samples: HashMap<AcousticGroup, Vec<(Vec<f32>, String)>> = HashMap::new();

    println!("\nGrouping samples...");
    for sample in &manifest.samples {
        if sample.labels.output == "None" {
            continue;
        }

        let acoustic_group = map_species_to_acoustic_group(&sample.labels.output);

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            group_samples
                                .entry(acoustic_group)
                                .or_default()
                                .push((features, sample.labels.output.clone()));
                        }
                    }
                }
            }
        }
    }

    // Convert to RawDataset
    let mut datasets: HashMap<AcousticGroup, RawDataset> = HashMap::new();

    for (group, samples) in group_samples {
        if samples.is_empty() {
            continue;
        }

        let n_samples = samples.len();
        let mut features = Array2::zeros((n_samples, FEATURE_DIM));
        let mut labels = Vec::new();
        let mut label_to_idx: HashMap<String, usize> = HashMap::new();
        let mut idx_to_label: HashMap<usize, String> = HashMap::new();

        for (i, (feat, label)) in samples.into_iter().enumerate() {
            for k in 0..FEATURE_DIM {
                features[[i, k]] = feat[k];
            }
            labels.push(label.clone());

            if !label_to_idx.contains_key(&label) {
                let idx = label_to_idx.len();
                label_to_idx.insert(label.clone(), idx);
                idx_to_label.insert(idx, label);
            }
        }

        datasets.insert(
            group,
            RawDataset {
                features,
                labels,
                label_to_idx,
                idx_to_label,
            },
        );
    }

    Ok(datasets)
}

// =============================================================================
// Training
// =============================================================================

fn train_specialist(group: AcousticGroup, dataset: &RawDataset) -> Result<RandomForestClassifier> {
    println!("\n========================================");
    println!("Training: {:?}", group);
    println!("========================================");
    println!("  Samples: {}", dataset.len());
    println!("  Classes: {}", dataset.num_classes());

    if dataset.num_classes() < 2 {
        println!("  SKIPPING: Only 1 class");
        anyhow::bail!("Insufficient classes");
    }

    let (train, test) = dataset.train_test_split(0.2, 42);
    println!("  Train: {}, Test: {}", train.len(), test.len());

    println!("  Training RF with {} trees...", N_ESTIMATORS);
    let start = Instant::now();

    let mut rf = RandomForestClassifier::new(N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT).with_balanced_weights();

    rf.fit_raw(&train.features, &train.labels, &train.label_to_idx, &train.idx_to_label)?;

    println!("  Training time: {:.2}s", start.elapsed().as_secs_f32());

    // Validate
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
        correct as f64 / test.len() as f64
    } else {
        0.0
    };

    println!("  Validation Accuracy: {:.2}%", accuracy * 100.0);

    Ok(rf)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("==============================================================");
    println!("  Acoustic Specialist RF Training");
    println!("  Using Acoustic Coherence instead of Biological Taxonomy");
    println!("==============================================================");
    println!();
    println!("Acoustic Splits:");
    println!("  Mammals:   Ultrasonic (bats) / Sonic_Long (whales) / Sonic_Short (primates)");
    println!("  Insects:   Wingbeat (mosquitoes) / Stridulation (crickets)");
    println!("  Birds:     High_Freq (songbirds) / Low_Freq (doves) / Mechanical");
    println!("  Marine:    Whistle (dolphins) / Click (porpoises) / Moan (whales)");
    println!();

    let datasets = load_data()?;

    println!("\nDataset sizes by acoustic group:");
    let mut groups: Vec<_> = datasets.iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (group, dataset) in &groups {
        println!(
            "  {:<25} {:>6} samples, {:>4} classes",
            format!("{:?}", group),
            dataset.len(),
            dataset.num_classes()
        );
    }

    // Collect group keys before moving datasets to Arc
    let group_keys: Vec<AcousticGroup> = groups.iter().map(|(g, _)| **g).collect();

    // Train specialists
    let output_dir = Path::new("specialist_rf_models");
    fs::create_dir_all(output_dir)?;

    println!("\nTraining specialists...");
    let start = Instant::now();

    let datasets_arc = Arc::new(datasets);
    let results: Vec<_> = group_keys
        .par_iter()
        .filter_map(|group| {
            let datasets = Arc::clone(&datasets_arc);

            datasets
                .get(group)
                .and_then(|dataset| train_specialist(*group, dataset).ok().map(|rf| (*group, rf)))
        })
        .collect();

    println!("\nTraining complete in {:.2}s", start.elapsed().as_secs_f32());

    // Save models in bincode format (much smaller and faster to load)
    println!("\nSaving models...");
    for (group, rf) in results {
        let filename = format!("specialist_rf_acoustic_{}.bincode", acoustic_group_name(group));
        let output_path = output_dir.join(&filename);

        let file = fs::File::create(&output_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &rf)?;

        println!("  Saved: {}", output_path.display());
    }

    println!("\nDone!");
    Ok(())
}
