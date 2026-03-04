//! BEANS-Zero Linguistic Profiling Pipeline
//! ==========================================
//!
//! Generates a "Bio-Acoustic Ethogram" - a map of communication styles across
//! the animal kingdom by analyzing the linguistic profile of each audio sample.
//!
//! ## Key Discovery: Duration CV as Linguistic Complexity Proxy
//!
//! The Duration Coefficient of Variation (CV) reveals the underlying communication type:
//! - **Low CV (~0.26-0.30)**: Uniform segments = Crystallized Song (songbirds, insects)
//! - **High CV (~0.40-0.95)**: Variable segments = Graded Calls (primates, bats)
//!
//! Usage:
//!   cargo run --release --bin profile_linguistics -- beans_zero_cache/beans_audio_manifest.json

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use technical_architecture::{LinguisticProfile, SmartSegmenter};

// Helper to convert LinguisticProfile to string key
fn profile_to_string(p: LinguisticProfile) -> String {
    match p {
        LinguisticProfile::CrystallizedSong => "CrystallizedSong".to_string(),
        LinguisticProfile::GradedCall => "GradedCall".to_string(),
        LinguisticProfile::CulturalCoda => "CulturalCoda".to_string(),
        LinguisticProfile::Mixed => "Mixed".to_string(),
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansSample {
    audio_file: String,
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    source_dataset: Option<String>,
    output: Option<String>,
    task: Option<String>,
    dataset_name: Option<String>,
    instruction_text: Option<String>,
}

// ============================================================================
// Results Structures
// ============================================================================

#[derive(Debug, Serialize)]
struct LinguisticProfileResults {
    total_samples: usize,
    profile_distribution: HashMap<String, usize>,
    dataset_composition: HashMap<String, ProfileStats>,
    top_species_by_profile: HashMap<String, Vec<SpeciesCount>>,
    audio_file_profiles: Vec<AudioFileProfile>,
}

#[derive(Debug, Serialize)]
struct ProfileStats {
    crystallized: usize,
    graded: usize,
    cultural: usize,
    mixed: usize,
    crystallized_pct: f64,
    graded_pct: f64,
}

#[derive(Debug, Serialize)]
struct SpeciesCount {
    species: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct AudioFileProfile {
    audio_file: String,
    dataset: String,
    species: String,
    profile: String,
    confidence: f32,
    energy_variance: f32,
    rhythmicity: f32,
    spectral_stability: f32,
    duration_cv: f32,
}

// ============================================================================
// Audio Processing
// ============================================================================

fn load_raw_audio(path: &Path, expected_samples: u32) -> Result<Vec<f32>> {
    let mut file = fs::File::open(path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;
    Ok(buffer
        .chunks_exact(4)
        .take(expected_samples as usize)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    let base_path = manifest_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║              BEANS-ZERO LINGUISTIC PROFILING PIPELINE                  ║");
    println!("║                    Bio-Acoustic Ethogram Generator                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load manifest
    println!("Loading manifest from: {:?}", manifest_path);
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    println!(
        "Dataset: {} ({} samples)",
        manifest.dataset, manifest.n_samples
    );

    // Initialize segmenter
    let mut segmenter = SmartSegmenter::new(44100);

    // Stats accumulators
    let mut profile_distribution: HashMap<String, usize> = HashMap::new();
    let mut dataset_stats: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut species_stats: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut audio_file_profiles: Vec<AudioFileProfile> = Vec::new();

    println!("\n{}", "=".repeat(70));
    println!("ANALYZING LINGUISTIC PROFILES");
    println!("{}", "=".repeat(70));

    let start_time = Instant::now();
    let mut processed = 0;
    let mut errors = 0;

    for sample in &manifest.samples {
        // Load audio
        let audio = match load_raw_audio(&base_path.join(&sample.audio_file), sample.n_samples) {
            Ok(a) => a,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        // Skip very short or silent audio
        let rms: f32 =
            (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        if rms < 0.001 || audio.len() < 4410 {
            errors += 1;
            continue;
        }

        // Analyze profile using SmartSegmenter
        let result = segmenter.segment_smart(&audio);
        let profile = result.analysis.profile;
        let profile_str = profile_to_string(profile);
        let confidence = result.analysis.confidence;

        // Get dataset name
        let dataset_name = sample
            .labels
            .dataset_name
            .clone()
            .or_else(|| sample.labels.source_dataset.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Get species name (from output or instruction)
        let species = sample
            .labels
            .output
            .clone()
            .or_else(|| sample.labels.instruction_text.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Update profile distribution
        *profile_distribution.entry(profile_str.clone()).or_insert(0) += 1;

        // Update dataset stats
        *dataset_stats
            .entry(dataset_name.clone())
            .or_default()
            .entry(profile_str.clone())
            .or_insert(0) += 1;

        // Update species stats
        *species_stats
            .entry(species.clone())
            .or_default()
            .entry(profile_str.clone())
            .or_insert(0) += 1;

        // Record audio file profile
        audio_file_profiles.push(AudioFileProfile {
            audio_file: sample.audio_file.clone(),
            dataset: dataset_name,
            species,
            profile: format!("{:?}", profile),
            confidence,
            energy_variance: result.analysis.energy_variance,
            rhythmicity: result.analysis.rhythmicity,
            spectral_stability: result.analysis.spectral_stability,
            duration_cv: result.analysis.estimated_duration_cv,
        });

        processed += 1;

        // Progress update
        if processed % 1000 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = processed as f64 / elapsed;
            let remaining = (manifest.n_samples - processed) as f64 / rate;
            println!(
                "  Processed {}/{} ({:.1}%) - {:.0} files/s - ETA: {:.0}s",
                processed,
                manifest.n_samples,
                processed as f64 / manifest.n_samples as f64 * 100.0,
                rate,
                remaining
            );
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!(
        "\nCompleted: {} samples processed, {} errors in {:.1}s",
        processed, errors, elapsed
    );

    // Compute dataset composition stats
    let mut dataset_composition: HashMap<String, ProfileStats> = HashMap::new();
    for (dataset, counts) in &dataset_stats {
        let total: usize = counts.values().sum();
        let crystallized = *counts.get("CrystallizedSong").unwrap_or(&0);
        let graded = *counts.get("GradedCall").unwrap_or(&0);
        let cultural = *counts.get("CulturalCoda").unwrap_or(&0);
        let mixed = *counts.get("Mixed").unwrap_or(&0);

        dataset_composition.insert(
            dataset.clone(),
            ProfileStats {
                crystallized,
                graded,
                cultural,
                mixed,
                crystallized_pct: crystallized as f64 / total.max(1) as f64 * 100.0,
                graded_pct: graded as f64 / total.max(1) as f64 * 100.0,
            },
        );
    }

    // Compute top species by profile
    let mut top_species_by_profile: HashMap<String, Vec<SpeciesCount>> = HashMap::new();

    for profile_key in &["CrystallizedSong", "GradedCall", "CulturalCoda", "Mixed"] {
        let mut ranked: Vec<_> = species_stats
            .iter()
            .filter_map(|(species, counts)| {
                let count = counts.get(*profile_key).copied().unwrap_or(0);
                if count > 0 {
                    Some((species.clone(), count))
                } else {
                    None
                }
            })
            .collect();

        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        top_species_by_profile.insert(
            profile_key.to_string(),
            ranked
                .into_iter()
                .take(10)
                .map(|(species, count)| SpeciesCount { species, count })
                .collect(),
        );
    }

    // Print results
    println!("\n{}", "=".repeat(70));
    println!("LINGUISTIC PROFILE DISTRIBUTION");
    println!("{}", "=".repeat(70));

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                   OVERALL PROFILE DISTRIBUTION                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");

    for profile in &["CrystallizedSong", "GradedCall", "CulturalCoda", "Mixed"] {
        let count = profile_distribution.get(*profile).copied().unwrap_or(0);
        let pct = count as f64 / processed.max(1) as f64 * 100.0;
        println!(
            "║  {:<20} {:>8} samples ({:>5.1}%)                       ║",
            profile, count, pct
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                TOP SPECIES BY LINGUISTIC PROFILE                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    print_top_species(
        &top_species_by_profile,
        "CrystallizedSong",
        "Crystallized Song (Birds/Insects) - Rhythmic, Stereotyped",
    );
    print_top_species(
        &top_species_by_profile,
        "GradedCall",
        "Graded Call (Mammals/Primates) - Variable Duration, Conversational",
    );
    print_top_species(
        &top_species_by_profile,
        "CulturalCoda",
        "Cultural Coda (Cetaceans) - Rhythmic but Learned",
    );

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                  DATASET LINGUISTIC COMPOSITION                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Dataset          | Crystallized | Graded   | Cultural | Mixed       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");

    // Sort datasets alphabetically
    let mut datasets: Vec<_> = dataset_composition.iter().collect();
    datasets.sort_by_key(|(k, _)| *k);

    for (dataset, stats) in datasets {
        let dataset_display = if dataset.len() > 16 {
            format!("{}...", &dataset[..13])
        } else {
            dataset.clone()
        };
        println!(
            "║  {:<16} | {:>5.1}%      | {:>5.1}%   | {:>5.1}%   | {:>5.1}%      ║",
            dataset_display,
            stats.crystallized_pct,
            stats.graded_pct,
            stats.cultural as f64
                / (stats.crystallized + stats.graded + stats.cultural + stats.mixed).max(1) as f64
                * 100.0,
            stats.mixed as f64
                / (stats.crystallized + stats.graded + stats.cultural + stats.mixed).max(1) as f64
                * 100.0
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    println!("\n{}", "=".repeat(70));
    println!("HYPOTHESIS VALIDATION");
    println!("{}", "=".repeat(70));
    println!();

    // Check hypothesis: Bird datasets should be Crystallized
    println!("Expected: Bird datasets (cbi, enabirds, zf-indiv) → Crystallized");
    for dataset in &["cbi", "enabirds", "zf-indiv", "esc50"] {
        if let Some(stats) = dataset_composition.get(*dataset) {
            let dominant = if stats.crystallized_pct > stats.graded_pct {
                "Crystallized"
            } else {
                "Graded"
            };
            println!(
                "  {}: {:.1}% Crystallized, {:.1}% Graded → {}",
                dataset, stats.crystallized_pct, stats.graded_pct, dominant
            );
        }
    }

    println!("\nExpected: Mammal datasets (watkins, humbugdb) → Graded/Mixed");
    for dataset in &["watkins", "humbugdb", "gibbons"] {
        if let Some(stats) = dataset_composition.get(*dataset) {
            let dominant = if stats.graded_pct > stats.crystallized_pct {
                "Graded"
            } else {
                "Crystallized"
            };
            println!(
                "  {}: {:.1}% Crystallized, {:.1}% Graded → {}",
                dataset, stats.crystallized_pct, stats.graded_pct, dominant
            );
        }
    }

    // Save results to JSON
    let results = LinguisticProfileResults {
        total_samples: processed,
        profile_distribution,
        dataset_composition,
        top_species_by_profile,
        audio_file_profiles,
    };

    let results_path = base_path.join("linguistic_profile_results.json");
    let file = fs::File::create(&results_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &results)?;
    println!("\nResults saved to: {:?}", results_path);

    Ok(())
}

fn print_top_species(top_species: &HashMap<String, Vec<SpeciesCount>>, profile: &str, title: &str) {
    println!("\n--- {} ---", title);

    if let Some(species_list) = top_species.get(profile) {
        for sc in species_list.iter().take(5) {
            println!("  {:<30} : {} calls", sc.species, sc.count);
        }
        if species_list.is_empty() {
            println!("  (No species with this profile)");
        }
    }
}
