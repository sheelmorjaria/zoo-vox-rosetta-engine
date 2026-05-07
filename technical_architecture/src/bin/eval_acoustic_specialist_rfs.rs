//! Evaluate Acoustic Specialist RF Models
//! ==============================================================
//!
//! This script evaluates the acoustic specialist RF models that were
//! trained using acoustic coherence instead of biological taxonomy.
//!
//! Usage:
//!   cargo run --release --bin eval_acoustic_specialist_rfs

use anyhow::Result;
use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use technical_architecture::classical_ml::RandomForestClassifier;

/// Load RF model from bincode (preferred) or JSON (fallback)
fn load_model(models_dir: &Path, group_name: &str) -> Result<RandomForestClassifier> {
    let bincode_path = models_dir.join(format!("specialist_rf_acoustic_{}.bincode", group_name));
    let json_path = models_dir.join(format!("specialist_rf_acoustic_{}.json", group_name));

    // Try bincode first (much faster and smaller)
    if bincode_path.exists() {
        let file = fs::File::open(&bincode_path)?;
        let reader = BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
        return Ok(model);
    }

    // Fallback to JSON
    if json_path.exists() {
        let data = fs::read_to_string(&json_path)?;
        let model = serde_json::from_str(&data)?;
        return Ok(model);
    }

    anyhow::bail!("No model file found for {}", group_name)
}

// =============================================================================
// Constants
// =============================================================================

const FEATURE_DIM: usize = 112;

// =============================================================================
// Acoustic Groups (same as train_acoustic_specialist_rfs.rs)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcousticGroup {
    // Mammals (3-way split)
    UltrasonicMammal,
    SonicLongMammal,
    SonicShortMammal,

    // Insects (2-way split)
    InsectWingbeat,
    InsectStridulation,

    // Birds (3-way split)
    BirdHighFreq,
    BirdLowFreq,
    BirdMechanical,

    // Marine Mammals (3-way split)
    MarineWhistle,
    MarineClick,
    MarineMoan,

    // Other
    Amphibian,
    Pinniped,
}

fn map_species_to_acoustic_group(species: &str) -> AcousticGroup {
    let s = species.to_lowercase();

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

    if s.contains("whale") && !s.contains("killer") {
        return AcousticGroup::MarineMoan;
    }

    if s.contains("seal")
        || s.contains("sea lion")
        || s.contains("walrus")
        || s.contains("phocid")
        || s.contains("otariid")
        || s.contains("otary")
    {
        return AcousticGroup::Pinniped;
    }

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

    if s.contains("hummingbird")
        || s.contains("trochilid")
        || s.contains("snipe")
        || s.contains("gallinago")
        || s.contains("woodpecker")
        || s.contains("picid")
    {
        return AcousticGroup::BirdMechanical;
    }

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
    rf_correct: usize,
    top5_correct: usize,
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
            rf_correct: 0,
            top5_correct: 0,
            task_type: "unknown".to_string(),
        }
    }
}

// =============================================================================
// Test Sample with Task Info
// =============================================================================

struct TestSample {
    features: Vec<f32>,
    label: String,
    task: String,
    acoustic_group: AcousticGroup,
}

// =============================================================================
// Data Loading
// =============================================================================

fn load_test_samples(test_ratio: f32, seed: u64) -> Result<Vec<TestSample>> {
    println!("Loading manifest...");
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples: {}", manifest.samples.len());

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_data = fs::read_to_string(cache_dir.join("cache_manifest.json"))?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features: {}", cache_manifest.entries.len());

    let mut all_samples: Vec<TestSample> = Vec::new();

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
                            all_samples.push(TestSample {
                                features,
                                label: sample.labels.output.clone(),
                                task: sample.labels.task.clone(),
                                acoustic_group,
                            });
                        }
                    }
                }
            }
        }
    }

    println!("  Loaded {} samples", all_samples.len());

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    all_samples.shuffle(&mut rng);

    let n_test = (all_samples.len() as f32 * test_ratio) as usize;
    Ok(all_samples.into_iter().take(n_test).collect())
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("==============================================================");
    println!("   Acoustic Specialist RF Evaluation");
    println!("==============================================================");
    println!();
    println!("Acoustic Groups:");
    println!("  Mammals:   Ultrasonic (bats) / Sonic_Long (whales) / Sonic_Short (primates)");
    println!("  Insects:   Wingbeat (mosquitoes) / Stridulation (crickets)");
    println!("  Birds:     High_Freq (songbirds) / Low_Freq (doves) / Mechanical");
    println!("  Marine:    Whistle (dolphins) / Click (porpoises) / Moan (whales)");
    println!();

    // Load test samples
    let test_samples = load_test_samples(0.2, 42)?;

    // Count by acoustic group
    let mut group_counts: HashMap<AcousticGroup, usize> = HashMap::new();
    for sample in &test_samples {
        *group_counts.entry(sample.acoustic_group).or_insert(0) += 1;
    }

    println!("\nTest samples by acoustic group:");
    let mut groups: Vec<_> = group_counts.iter().collect();
    groups.sort_by(|a, b| b.1.cmp(a.1));
    for (group, count) in &groups {
        println!("  {:<25} {} samples", format!("{:?}", group), count);
    }

    // Load RF specialists
    println!("\nLoading RF Specialists...");
    let models_dir = Path::new("specialist_rf_models");
    let mut rf_specialists: HashMap<AcousticGroup, RandomForestClassifier> = HashMap::new();

    for group in [
        AcousticGroup::UltrasonicMammal,
        AcousticGroup::SonicLongMammal,
        AcousticGroup::SonicShortMammal,
        AcousticGroup::InsectWingbeat,
        AcousticGroup::InsectStridulation,
        AcousticGroup::BirdHighFreq,
        AcousticGroup::BirdLowFreq,
        AcousticGroup::BirdMechanical,
        AcousticGroup::MarineWhistle,
        AcousticGroup::MarineClick,
        AcousticGroup::MarineMoan,
        AcousticGroup::Amphibian,
        AcousticGroup::Pinniped,
    ] {
        let group_name = acoustic_group_name(group);
        let bincode_path = models_dir.join(format!("specialist_rf_acoustic_{}.bincode", group_name));
        let json_path = models_dir.join(format!("specialist_rf_acoustic_{}.json", group_name));

        if bincode_path.exists() || json_path.exists() {
            match load_model(models_dir, group_name) {
                Ok(model) => {
                    let format = if bincode_path.exists() { "bincode" } else { "JSON" };
                    println!("  {:?}: {} classes ({})", group, model.n_classes(), format);
                    rf_specialists.insert(group, model);
                }
                Err(e) => {
                    println!("  {:?}: FAILED to load - {}", group, e);
                }
            }
        }
    }

    // =========================================================================
    // Evaluation
    // =========================================================================
    println!("\n==============================================================");
    println!("  RF-Only Classification Pipeline");
    println!("==============================================================");

    let mut total_correct = 0usize;
    let mut total_top5_correct = 0usize;
    let total_samples = test_samples.len();
    let mut dataset_metrics: HashMap<String, DatasetMetrics> = HashMap::new();

    for sample in &test_samples {
        let acoustic_group = sample.acoustic_group;

        // Get metrics for this dataset
        let metrics = dataset_metrics
            .entry(sample.task.clone())
            .or_insert_with(|| DatasetMetrics {
                task_type: sample.task.clone(),
                ..Default::default()
            });
        metrics.samples += 1;

        // RF prediction
        if let Some(rf_model) = rf_specialists.get(&acoustic_group) {
            let features_arr = Array1::from_vec(sample.features.clone());
            let pred_idx = rf_model.predict(&features_arr);

            // Top-1 check
            if let Some(pred_label) = rf_model.idx_to_label().get(&pred_idx) {
                if pred_label == &sample.label {
                    total_correct += 1;
                    metrics.rf_correct += 1;
                }
            }

            // Top-5 check
            let probs = rf_model.predict_proba(&features_arr);
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            #[allow(clippy::unnecessary_sort_by)]
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, _) in indexed_probs.iter().take(5) {
                if let Some(pred_label) = rf_model.idx_to_label().get(idx) {
                    if pred_label == &sample.label {
                        total_top5_correct += 1;
                        metrics.top5_correct += 1;
                        break;
                    }
                }
            }
        }
    }

    // Calculate final metrics per dataset
    for metrics in dataset_metrics.values_mut() {
        let n = metrics.samples as f64;
        metrics.top1_accuracy = metrics.rf_correct as f64 / n;
        metrics.top5_accuracy = metrics.top5_correct as f64 / n;
        metrics.accuracy = metrics.top1_accuracy;
        metrics.ensemble_accuracy = metrics.top1_accuracy;
        metrics.precision = metrics.top1_accuracy;
        metrics.recall = metrics.top1_accuracy;
        metrics.f1_score = metrics.top1_accuracy;
    }

    let overall_accuracy = if total_samples > 0 {
        total_correct as f64 / total_samples as f64
    } else {
        0.0
    };

    let overall_top5 = if total_samples > 0 {
        total_top5_correct as f64 / total_samples as f64
    } else {
        0.0
    };

    println!();
    println!("Model Performance:");
    println!("-----------------");
    println!(
        "  RF Top-1 accuracy:  {:>6.2}%  ({}/{})",
        overall_accuracy * 100.0,
        total_correct,
        total_samples
    );
    println!(
        "  RF Top-5 accuracy:  {:>6.2}%  ({}/{})",
        overall_top5 * 100.0,
        total_top5_correct,
        total_samples
    );

    // Per-dataset breakdown
    println!();
    println!("--- Per-Dataset Breakdown ---");
    let mut datasets: Vec<_> = dataset_metrics.iter().collect();
    datasets.sort_by_key(|b| std::cmp::Reverse(b.1.samples));

    println!("{:<25} {:>6} {:>8} {:>10}", "Dataset", "n", "Top-1", "Top-5");
    println!("{}", "-".repeat(52));

    for (dataset, metrics) in datasets.iter() {
        println!(
            "{:<25} {:>6} {:>7.1}% {:>9.1}%",
            dataset,
            metrics.samples,
            metrics.top1_accuracy * 100.0,
            metrics.top5_accuracy * 100.0
        );
    }

    // Save results to JSON
    let results_output = serde_json::to_string_pretty(&dataset_metrics)?;
    std::fs::write("acoustic_specialist_rf_results.json", &results_output)?;
    println!("\nDetailed results saved to: acoustic_specialist_rf_results.json");

    // =========================================================================
    // Summary
    // =========================================================================
    println!();
    println!("==============================================================");
    println!("  Evaluation Summary");
    println!("==============================================================");
    println!("  RF Top-1:          {:>6.2}%", overall_accuracy * 100.0);
    println!("  RF Top-5:          {:>6.2}%", overall_top5 * 100.0);
    println!("==============================================================");

    println!("\nDone!");
    Ok(())
}
