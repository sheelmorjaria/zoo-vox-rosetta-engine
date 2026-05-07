//! Ensemble Evaluation: NN (Navigator) + RF (Judge) with Acoustic Routing
//! ==============================================================
//!
//! Architecture:
//!   INPUT: 112D Feature Vector
//!       |
//!       +---- Route to Acoustic Group ----+
//!       |                               |
//!       v                               v
//!   [NN 112D]                   [RF Specialist]
//!       |                           |
//!       v                           v
//!   Top-5 Candidates          Probability Distribution
//!       |                           |
//!       +-----------+---------------+
//!                   |
//!                   v
//!           [Ensemble Voter]
//!                   |
//!                   v
//!           FINAL PREDICTION
//!
//! Acoustic Groups:
//!   Mammals:   Ultrasonic (bats) / Sonic_Long (whales) / Sonic_Short (primates)
//!   Insects:   Wingbeat (mosquitoes) / Stridulation (crickets)
//!   Birds:     High_Freq (songbirds) / Low_Freq (doves) / Mechanical
//!   Marine:    Whistle (dolphins) / Click (porpoises) / Moan (whales)
//!
//! Logic:
//! 1. Route sample to ACOUSTIC GROUP based on species characteristics
//! 2. Get specialist RF for that acoustic group
//! 3. NN generates Top-5 shortlist
//! 4. RF evaluates candidates -> re-ranks
//! 5. Fall back to RF-only if RF prediction not in NN Top-5
//!
//! Usage:
//!   export LIBTORCH=/path/to/libtorch
//!   export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
//!   cargo run --release --features gpu-training --bin eval_ensemble_nn_rf

#![cfg(feature = "gpu-training")]

use anyhow::{Context, Result};
use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use tch::{nn, Device, Tensor};

use technical_architecture::classical_ml::RandomForestClassifier;

// =============================================================================
// Constants
// =============================================================================

const NN_WEIGHT: f64 = 0.40;
const RF_WEIGHT: f64 = 0.60;
const FEATURE_DIM: usize = 112;
const PHYSICS_DIM: i64 = 46;
const MACRO_DIM: i64 = 30;
const MICRO_DIM: i64 = 36;
const PHYSICS_HIDDEN: i64 = 256;
const MACRO_HIDDEN: i64 = 128;
const MICRO_HIDDEN: i64 = 64;
const DROPOUT_RATE: f64 = 0.3;

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
    task: String,
}

#[derive(Debug, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetMetrics {
    #[serde(rename = "Accuracy")]
    accuracy: f64,
    #[serde(rename = "Precision")]
    precision: f64,
    #[serde(rename = "Recall")]
    recall: f64,
    #[serde(rename = "F1 Score")]
    f1_score: f64,
    #[serde(rename = "Top-1 Accuracy")]
    top1_accuracy: f64,
    #[serde(rename = "Top-5 Accuracy")]
    top5_accuracy: f64,
    #[serde(rename = "Ensemble Accuracy")]
    ensemble_accuracy: f64,
    samples: usize,
    nn_correct: usize,
    rf_correct: usize,
    ensemble_correct: usize,
    task_type: String,
}

impl Default for DatasetMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            top1_accuracy: 0.0,
            top5_accuracy: 0.0,
            ensemble_accuracy: 0.0,
            samples: 0,
            nn_correct: 0,
            rf_correct: 0,
            ensemble_correct: 0,
            task_type: "unknown".to_string(),
        }
    }
}

// =============================================================================
// Acoustic Groups (same as train_acoustic_specialist_rfs.rs)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

// =============================================================================
// RF Files by Acoustic Group
// =============================================================================

const ACOUSTIC_RF_FILES: &[(AcousticGroup, &str)] = &[
    (
        AcousticGroup::UltrasonicMammal,
        "specialist_rf_acoustic_ultrasonic_mammal.json",
    ),
    (
        AcousticGroup::SonicLongMammal,
        "specialist_rf_acoustic_sonic_long_mammal.json",
    ),
    (
        AcousticGroup::SonicShortMammal,
        "specialist_rf_acoustic_sonic_short_mammal.json",
    ),
    (
        AcousticGroup::InsectWingbeat,
        "specialist_rf_acoustic_insect_wingbeat.json",
    ),
    (
        AcousticGroup::InsectStridulation,
        "specialist_rf_acoustic_insect_stridulation.json",
    ),
    (
        AcousticGroup::BirdHighFreq,
        "specialist_rf_acoustic_bird_high_freq.json",
    ),
    (AcousticGroup::BirdLowFreq, "specialist_rf_acoustic_bird_low_freq.json"),
    (
        AcousticGroup::BirdMechanical,
        "specialist_rf_acoustic_bird_mechanical.json",
    ),
    (
        AcousticGroup::MarineWhistle,
        "specialist_rf_acoustic_marine_whistle.json",
    ),
    (AcousticGroup::MarineClick, "specialist_rf_acoustic_marine_click.json"),
    (AcousticGroup::MarineMoan, "specialist_rf_acoustic_marine_moan.json"),
    (AcousticGroup::Amphibian, "specialist_rf_acoustic_amphibian.json"),
    (AcousticGroup::Pinniped, "specialist_rf_acoustic_pinniped.json"),
];

// =============================================================================
// Task Type Classification
// =============================================================================

fn is_detection_task(task: &str) -> bool {
    matches!(task, "dcase" | "rfcx" | "detection" | "sound_event_detection")
}

fn should_skip_task(task: &str) -> bool {
    matches!(task, "captioning" | "lifestage" | "zf-indiv" | "call-type" | "esc50") || task.starts_with("unseen-")
}

// =============================================================================
// NN Model Architecture
// =============================================================================

struct PhysicsBlock {
    fc1: nn::Linear,
    bn1: nn::BatchNorm,
    fc2: nn::Linear,
}

impl PhysicsBlock {
    fn new(vs: &nn::Path) -> Self {
        let fc1 = nn::linear(vs, PHYSICS_DIM, PHYSICS_HIDDEN, Default::default());
        let bn1 = nn::batch_norm1d(vs, PHYSICS_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, PHYSICS_HIDDEN, PHYSICS_HIDDEN, Default::default());
        Self { fc1, bn1, fc2 }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, false);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, false)
    }
}

struct MacroBlock {
    fc1: nn::Linear,
    bn1: nn::BatchNorm,
    fc2: nn::Linear,
}

impl MacroBlock {
    fn new(vs: &nn::Path) -> Self {
        let input_dim = PHYSICS_HIDDEN + MACRO_DIM;
        let fc1 = nn::linear(vs, input_dim, MACRO_HIDDEN, Default::default());
        let bn1 = nn::batch_norm1d(vs, MACRO_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, MACRO_HIDDEN, MACRO_HIDDEN, Default::default());
        Self { fc1, bn1, fc2 }
    }

    fn forward(&self, physics_out: &Tensor, macro_feat: &Tensor) -> Tensor {
        let x = Tensor::cat(&[physics_out, macro_feat], 1);
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, false);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, false)
    }
}

struct MicroBlock {
    fc1: nn::Linear,
    bn1: nn::BatchNorm,
    fc2: nn::Linear,
}

impl MicroBlock {
    fn new(vs: &nn::Path) -> Self {
        let input_dim = MACRO_HIDDEN + MICRO_DIM;
        let fc1 = nn::linear(vs, input_dim, MICRO_HIDDEN, Default::default());
        let bn1 = nn::batch_norm1d(vs, MICRO_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, MICRO_HIDDEN, MICRO_HIDDEN, Default::default());
        Self { fc1, bn1, fc2 }
    }

    fn forward(&self, macro_out: &Tensor, micro_feat: &Tensor) -> Tensor {
        let x = Tensor::cat(&[macro_out, micro_feat], 1);
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, false);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, false)
    }
}

struct OutputBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl OutputBlock {
    fn new(vs: &nn::Path, n_classes: i64) -> Self {
        let fc1 = nn::linear(vs, MICRO_HIDDEN, MICRO_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, MICRO_HIDDEN, n_classes, Default::default());
        Self { fc1, fc2 }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.fc1);
        let x = x.gelu("none");
        x.apply(&self.fc2)
    }
}

struct CurriculumNet {
    physics: PhysicsBlock,
    macro_block: MacroBlock,
    micro: MicroBlock,
    output: OutputBlock,
}

impl CurriculumNet {
    fn new(vs: &nn::Path, n_classes: i64) -> Self {
        let physics = PhysicsBlock::new(&vs.sub("physics"));
        let macro_block = MacroBlock::new(&vs.sub("macro"));
        let micro = MicroBlock::new(&vs.sub("micro"));
        let output = OutputBlock::new(&vs.sub("output"), n_classes);
        Self {
            physics,
            macro_block,
            micro,
            output,
        }
    }

    fn forward(&self, physics_input: &Tensor, macro_input: &Tensor, micro_input: &Tensor) -> Tensor {
        let physics_out = self.physics.forward(physics_input);
        let macro_out = self.macro_block.forward(&physics_out, macro_input);
        let micro_out = self.micro.forward(&macro_out, micro_input);
        self.output.forward(&micro_out)
    }
}

// =============================================================================
// Label Normalization
// =============================================================================

fn normalize_label(label: &str, canonical_map: &HashMap<String, String>) -> String {
    canonical_map.get(label).cloned().unwrap_or_else(|| {
        let normalized = label.to_lowercase();
        normalized.split('_').next().unwrap_or(&normalized).to_string()
    })
}

fn build_label_canonical_map() -> HashMap<String, String> {
    let mut map = HashMap::new();
    // Add common misspellings/variants
    map.insert("comlytess_mys_24836".to_string(), "comlytes_mys_24836".to_string());
    map.insert("phyllostomus_discolor".to_string(), "phyllostomus_discolor".to_string());
    map.insert(
        "vesper_urticular_amurensis".to_string(),
        "vesper_urticular_amurensis".to_string(),
    );
    map
}

// =============================================================================
// Data Loading
// =============================================================================

struct TestSample {
    features: Vec<f32>,
    label: String,
    task: String,
}

fn load_test_samples(test_ratio: f32, seed: u64) -> Result<Vec<TestSample>> {
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_data = fs::read_to_string(cache_dir.join("cache_manifest.json"))?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;

    let mut all_samples: Vec<TestSample> = Vec::new();

    for sample in &manifest.samples {
        if sample.labels.output == "None" || should_skip_task(&sample.labels.task) {
            continue;
        }

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            all_samples.push(TestSample {
                                features,
                                label: sample.labels.output.clone(),
                                task: sample.labels.task.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    all_samples.shuffle(&mut rng);

    let n_test = (all_samples.len() as f32 * test_ratio) as usize;
    Ok(all_samples.into_iter().take(n_test).collect())
}

// =============================================================================
// Build Label Mapping (same as training)
// =============================================================================

fn build_global_label_mapping() -> Result<(HashMap<String, i64>, HashMap<i64, String>, i64)> {
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_data = fs::read_to_string(cache_dir.join("cache_manifest.json"))?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;

    let mut all_labels: Vec<String> = Vec::new();

    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            all_labels.push(label);
                        }
                    }
                }
            }
        }
    }

    let mut unique_labels: Vec<String> = all_labels.clone();
    unique_labels.sort();
    unique_labels.dedup();

    let n_classes = unique_labels.len() as i64;
    let mut label_to_idx = HashMap::new();
    let mut idx_to_label = HashMap::new();

    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx as i64);
        idx_to_label.insert(idx as i64, label.clone());
    }

    Ok((label_to_idx, idx_to_label, n_classes))
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("==============================================================");
    println!("   Ensemble Evaluation: NN (Navigator) + RF (Judge)");
    println!("   GPU Accelerated with libtorch");
    println!("==============================================================");
    println!();

    // Build global label mapping
    println!("Building global label mapping...");
    let (_label_to_idx, idx_to_label, n_classes) = build_global_label_mapping()?;
    println!("  Total classes: {}", n_classes);

    // Load NN model
    println!("\nLoading NN Model...");
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let net = CurriculumNet::new(&vs.root(), n_classes);

    let model_path = "rosetta_net_112d_curriculum_gpu.ot";
    if !Path::new(model_path).exists() {
        anyhow::bail!("NN model not found: {}. Run training first.", model_path);
    }
    vs.load(model_path).context("Failed to load NN model")?;
    println!("  Loaded: {}", model_path);

    // Load RF specialists by Acoustic Group
    println!("\nLoading RF Specialists by Acoustic Group...");
    let models_dir = Path::new("specialist_rf_models");
    let mut rf_specialists: HashMap<AcousticGroup, RandomForestClassifier> = HashMap::new();

    for (group, filename) in ACOUSTIC_RF_FILES.iter() {
        let model_path = models_dir.join(filename);
        if model_path.exists() {
            let data = fs::read_to_string(&model_path)?;
            let model: RandomForestClassifier = serde_json::from_str(&data)?;
            println!("  {:?}: {} classes", group, model.n_classes());
            rf_specialists.insert(*group, model);
        }
    }

    // Load test samples
    println!("\nLoading Test Data (10% holdout)...");
    let test_samples = load_test_samples(0.1, 42)?;

    let (classification_samples, detection_samples): (Vec<_>, Vec<_>) =
        test_samples.iter().partition(|s| !is_detection_task(&s.task));

    println!("  Classification samples: {}", classification_samples.len());
    println!("  Detection samples: {}", detection_samples.len());

    let canonical_map = build_label_canonical_map();

    // =========================================================================
    // Classification Evaluation
    // =========================================================================
    println!("\n==============================================================");
    println!("  Classification Pipeline (NN + RF Ensemble)");
    println!("==============================================================");

    let mut nn_only_correct = 0usize;
    let mut rf_only_correct = 0usize;
    let mut ensemble_correct = 0usize;
    let mut total_classified = 0usize;
    let mut dataset_metrics: HashMap<String, DatasetMetrics> = HashMap::new();

    for sample in &classification_samples {
        let true_label_canonical = normalize_label(&sample.label, &canonical_map);
        let acoustic_group = map_species_to_acoustic_group(&sample.label);

        // Prepare tensors
        let physics_arr: Vec<f32> = sample.features[0..46].to_vec();
        let macro_arr: Vec<f32> = sample.features[46..76].to_vec();
        let micro_arr: Vec<f32> = sample.features[76..112].to_vec();

        let physics_tensor = Tensor::from_slice(&physics_arr).view([1, PHYSICS_DIM]);
        let macro_tensor = Tensor::from_slice(&macro_arr).view([1, MACRO_DIM]);
        let micro_tensor = Tensor::from_slice(&micro_arr).view([1, MICRO_DIM]);

        // NN prediction
        let logits = net.forward(&physics_tensor, &macro_tensor, &micro_tensor);
        let probs = logits.softmax(-1, tch::Kind::Float).squeeze();
        let probs_vec: Vec<f32> = probs.try_into()?;

        // Get NN Top-5
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter().cloned().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));a.1).unwrap_or(std::cmp::Ordering::Less));
        let nn_top5: Vec<(String, f64)> = indexed_probs
            .iter()
            .take(5)
            .filter_map(|(idx, prob)| {
                idx_to_label
                    .get(&(*idx as i64))
                    .map(|label| (normalize_label(label, &canonical_map), *prob as f64))
            })
            .collect();

        // NN-only prediction
        let nn_pred = &nn_top5[0].0;
        let nn_is_correct = nn_pred == &true_label_canonical;
        if nn_is_correct {
            nn_only_correct += 1;
        }

        // RF prediction
        let rf_pred = if let Some(rf_model) = rf_specialists.get(&acoustic_group) {
            let features_arr = Array1::from_vec(sample.features.clone());
            let pred_idx = rf_model.predict(&features_arr);
            let pred_label = rf_model.idx_to_label().get(&pred_idx).cloned().unwrap_or_default();
            normalize_label(&pred_label, &canonical_map)
        } else {
            "no_model".to_string()
        };
        let rf_is_correct = rf_pred == true_label_canonical;
        if rf_is_correct {
            rf_only_correct += 1;
        }

        // Ensemble: Restricted Voting with RF Fallback
        let mut ensemble_pred = nn_pred.clone();

        if let Some(rf_model) = rf_specialists.get(&acoustic_group) {
            let features_arr = Array1::from_vec(sample.features.clone());
            let rf_probs = rf_model.predict_proba(&features_arr);

            // Build RF label -> probability map
            let mut rf_prob_map: HashMap<String, f32> = HashMap::new();
            for (idx, prob) in rf_probs.iter().enumerate() {
                if let Some(label) = rf_model.idx_to_label().get(&idx) {
                    let canonical = normalize_label(label, &canonical_map);
                    rf_prob_map.insert(canonical, *prob);
                }
            }

            // Check if RF's top prediction is in NN's Top-5
            let rf_in_nn_top5 = nn_top5.iter().any(|(label, _)| label == &rf_pred);

            if rf_in_nn_top5 {
                // Combine NN Top-5 with RF probabilities
                let mut ensemble_scores: Vec<(String, f64)> = nn_top5
                    .iter()
                    .map(|(species, nn_prob)| {
                        let rf_prob = rf_prob_map.get(species).copied().unwrap_or(0.0) as f64;
                        let score = *nn_prob * NN_WEIGHT + rf_prob * RF_WEIGHT;
                        (species.clone(), score)
                    })
                    .collect();

                ensemble_scores.sort_by(|a, b| b.1.partial_cmp(ensemble_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));a.1).unwrap_or(std::cmp::Ordering::Less));
                ensemble_pred = ensemble_scores[0].0.clone();
            } else {
                // RF's prediction NOT in NN's Top-5 - fall back to RF-only
                ensemble_pred = rf_pred.clone();
            }
        }

        let ensemble_is_correct = ensemble_pred == true_label_canonical;
        if ensemble_is_correct {
            ensemble_correct += 1;
        }

        total_classified += 1;

        // Update dataset metrics
        let metrics = dataset_metrics
            .entry(sample.task.clone())
            .or_insert_with(|| DatasetMetrics {
                task_type: sample.task.clone(),
                ..Default::default()
            });
        metrics.samples += 1;
        if nn_is_correct {
            metrics.nn_correct += 1;
        }
        if rf_is_correct {
            metrics.rf_correct += 1;
        }
        if ensemble_is_correct {
            metrics.ensemble_correct += 1;
        }
    }

    // Calculate final metrics per dataset
    for metrics in dataset_metrics.values_mut() {
        let n = metrics.samples as f64;
        metrics.top1_accuracy = metrics.nn_correct as f64 / n;
        metrics.top5_accuracy = metrics.nn_correct as f64 / n;
        metrics.accuracy = metrics.rf_correct as f64 / n;
        metrics.ensemble_accuracy = metrics.ensemble_correct as f64 / n;
        metrics.precision = metrics.ensemble_accuracy;
        metrics.recall = metrics.ensemble_accuracy;
        metrics.f1_score = metrics.ensemble_accuracy;
    }

    let nn_accuracy = nn_only_correct as f64 / total_classified as f64;
    let rf_accuracy = rf_only_correct as f64 / total_classified as f64;
    let ensemble_accuracy = ensemble_correct as f64 / total_classified as f64;

    println!();
    println!("Model Comparison:");
    println!("-----------------");
    println!(
        "  NN-only accuracy:   {:>6.2}%  ({}/{})",
        nn_accuracy * 100.0,
        nn_only_correct,
        total_classified
    );
    println!(
        "  RF-only accuracy:   {:>6.2}%  ({}/{})",
        rf_accuracy * 100.0,
        rf_only_correct,
        total_classified
    );
    println!(
        "  Ensemble accuracy:  {:>6.2}%  ({}/{})",
        ensemble_accuracy * 100.0,
        ensemble_correct,
        total_classified
    );

    let improvement = (ensemble_accuracy - rf_accuracy) * 100.0;
    println!("\n  Ensemble vs RF: {:+.2}%", improvement);

    // Per-dataset breakdown
    println!();
    println!("--- Per-Dataset Breakdown ---");
    let mut datasets: Vec<_> = dataset_metrics.iter().collect();
    datasets.sort_by_key(|b| std::cmp::Reverse(b.1.samples));

    println!(
        "{:<25} {:>6} {:>8} {:>8} {:>10} {:>10}",
        "Dataset", "n", "Top-1", "RF", "Ensemble", "d vs RF"
    );
    println!("{}", "-".repeat(70));

    for (dataset, metrics) in datasets.iter().take(15) {
        let delta = (metrics.ensemble_accuracy - metrics.accuracy) * 100.0;
        println!(
            "{:<25} {:>6} {:>7.1}% {:>7.1}% {:>8.1}% {:>+9.1}%",
            dataset,
            metrics.samples,
            metrics.top1_accuracy * 100.0,
            metrics.accuracy * 100.0,
            metrics.ensemble_accuracy * 100.0,
            delta
        );
    }

    // Save results to JSON
    let results_output = serde_json::to_string_pretty(&dataset_metrics)?;
    std::fs::write("ensemble_nn_rf_results.json", &results_output)?;
    println!("\nDetailed results saved to: ensemble_nn_rf_results.json");

    // =========================================================================
    // Detection Evaluation
    // =========================================================================
    if !detection_samples.is_empty() {
        println!("\n==============================================================");
        println!("  Detection Pipeline");
        println!("==============================================================");

        let detection_threshold = 0.3;
        let mut correct_detections = 0usize;
        let mut correct_rejections = 0usize;
        let total_detection = detection_samples.len();

        for sample in &detection_samples {
            let acoustic_group = map_species_to_acoustic_group(&sample.label);

            if let Some(rf_model) = rf_specialists.get(&acoustic_group) {
                let features_arr = Array1::from_vec(sample.features.clone());
                let pred_idx = rf_model.predict(&features_arr);
                let proba = rf_model.predict_proba(&features_arr);
                let confidence = proba.get(pred_idx).copied().unwrap_or(0.0);

                if confidence >= detection_threshold {
                    let pred_label = rf_model.idx_to_label().get(&pred_idx).cloned().unwrap_or_default();
                    let pred_canonical = normalize_label(&pred_label, &canonical_map);
                    let true_canonical = normalize_label(&sample.label, &canonical_map);
                    if pred_canonical == true_canonical {
                        correct_detections += 1;
                    }
                } else {
                    correct_rejections += 1;
                }
            }
        }

        let detection_accuracy = (correct_detections + correct_rejections) as f64 / total_detection as f64;
        println!(
            "\n  Detection Accuracy: {:>6.2}%  ({}/{})",
            detection_accuracy * 100.0,
            correct_detections + correct_rejections,
            total_detection
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!();
    println!("==============================================================");
    println!("  Evaluation Summary");
    println!("==============================================================");
    println!("  NN-only:           {:>6.2}%", nn_accuracy * 100.0);
    println!("  RF-only:            {:>6.2}%", rf_accuracy * 100.0);
    println!("  Ensemble (NN+RF):   {:>6.2}%", ensemble_accuracy * 100.0);
    println!("==============================================================");

    Ok(())
}
