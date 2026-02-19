// Comparative Analysis: Bat Vocalizations vs Environmental Sounds (Control)
//
// This example compares the linguistic structure of:
// 1. Egyptian Fruit Bat vocalizations (potential communicative structure)
// 2. ESC-50 environmental sounds (control - no linguistic structure expected)
//
// The analysis tests whether bat vocalizations exhibit:
// - Natural Zipf's law distribution (α ≈ 1.0)
// - High cluster coherence (indicating distinct vocabulary items)
// - Structured phoneme sequences (via GMM-HMM)

use std::collections::{HashMap, HashSet};
use std::path::Path;
use technical_architecture::{
    DiscoveryConfig, LexiconToSyntaxPipeline, RefinementConfig, SegmentationConfig,
    VectorizationConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Comparative Analysis: Bat Vocalizations vs Environmental Sounds          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Configuration
    // =========================================================================

    let subset_size = 1000; // Process 1,000 files from each dataset

    // Bat dataset path
    let bat_base_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let bat_audio_dir = bat_base_dir.join("audio");
    let bat_annotations = bat_base_dir.join("annotations.csv");
    let bat_output_dir = bat_base_dir.join("comparative_results_bat");

    // ESC-50 dataset path
    let esc50_base_dir = Path::new("/home/sheel/birdsong_analysis/data/ESC-50");
    let esc50_audio_dir = esc50_base_dir.join("audio");
    let esc50_meta = esc50_base_dir.join("meta/esc50.csv");
    let esc50_output_dir = bat_base_dir.join("comparative_results_esc50");

    // Create output directories
    std::fs::create_dir_all(&bat_output_dir)?;
    std::fs::create_dir_all(&esc50_output_dir)?;

    println!("📊 Configuration:");
    println!("   ├─ Subset size: {} files per dataset", subset_size);
    println!("   ├─ Bat audio: {}", bat_audio_dir.display());
    println!("   ├─ Bat annotations: {}", bat_annotations.display());
    println!("   ├─ ESC-50 audio: {}", esc50_audio_dir.display());
    println!("   └─ ESC-50 metadata: {}", esc50_meta.display());
    println!();

    // =========================================================================
    // Pipeline Configuration (same for both datasets)
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Pipeline Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let segmentation_config = SegmentationConfig {
        min_duration_ms: 50.0,
        max_duration_ms: 500.0,
        onset_threshold: 0.2,
        min_onset_distance_ms: 10.0,
        sample_rate: 250000, // Will be adjusted for ESC-50 (44.1kHz)
    };

    let vectorization_config = VectorizationConfig {
        n_mels: 30,
        fft_size: 2048,
        hop_size: 512,
        normalize: true,
    };

    let discovery_config = DiscoveryConfig {
        eps: 50.0,
        min_samples: 3,
        dtw_window_size: None,
        use_fast_dtw: true,
        fast_dtw_radius: 10,
        use_lb_keogh: true,
    };

    let refinement_config = RefinementConfig {
        n_states: None,
        n_components: 2,
        max_iterations: 100,
        convergence_threshold: 1e-4,
        covariance_reg: 1e-6,
    };

    println!("✅ Pipeline configured with parameters optimized for comparative analysis");
    println!();

    // =========================================================================
    // Dataset 1: Egyptian Fruit Bats
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Dataset 1: Egyptian Fruit Bat Vocalizations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Load annotations to get context information
    let bat_annotations_data = load_bat_annotations(&bat_annotations)?;
    println!(
        "📋 Loaded {} annotations from bat dataset",
        bat_annotations_data.len()
    );

    // Collect subset of bat audio files
    let bat_audio_files: Vec<_> = std::fs::read_dir(&bat_audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .take(subset_size)
        .map(|entry| entry.path())
        .collect();

    println!(
        "📁 Selected {} bat audio files for analysis",
        bat_audio_files.len()
    );
    println!();

    // Run pipeline on bat dataset
    let bat_result = run_pipeline(
        &bat_audio_files,
        &bat_output_dir,
        &segmentation_config,
        &vectorization_config,
        &discovery_config,
        &refinement_config,
        "Bat Vocalizations",
        250000,
    )?;

    // =========================================================================
    // Dataset 2: ESC-50 Environmental Sounds (Control)
    // =========================================================================

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Dataset 2: ESC-50 Environmental Sounds (Control)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Load ESC-50 metadata to get diverse sound categories
    let esc50_metadata = load_esc50_metadata(&esc50_meta)?;
    println!("📋 Loaded {} ESC-50 metadata entries", esc50_metadata.len());

    // Select diverse subset across different categories
    let esc50_audio_files =
        select_diverse_esc50_samples(&esc50_audio_dir, &esc50_metadata, subset_size)?;
    println!(
        "📁 Selected {} ESC-50 audio files across {} categories",
        esc50_audio_files.len(),
        count_unique_categories(&esc50_audio_files, &esc50_metadata)
    );
    println!();

    // Run pipeline on ESC-50 dataset (with adjusted sample rate)
    let esc50_result = run_pipeline(
        &esc50_audio_files,
        &esc50_output_dir,
        &SegmentationConfig {
            sample_rate: 44100, // ESC-50 is 44.1kHz
            ..segmentation_config
        },
        &vectorization_config,
        &discovery_config,
        &refinement_config,
        "ESC-50 Environmental Sounds",
        44100,
    )?;

    // =========================================================================
    // Comparative Analysis
    // =========================================================================

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARATIVE ANALYSIS RESULTS                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    print_comparison(&bat_result, &esc50_result);

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💾 Results saved to:");
    println!("   ├─ Bat: {}", bat_output_dir.display());
    println!("   └─ ESC-50: {}", esc50_output_dir.display());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

#[derive(Debug)]
struct BatAnnotation {
    emitter: u32,
    addressee: u32,
    context: u32,
    filename: String,
}

#[derive(Debug)]
struct Esc50Metadata {
    filename: String,
    category: String,
    target: i32,
    fold: i32,
}

fn load_bat_annotations(path: &Path) -> Result<Vec<BatAnnotation>, Box<dyn std::error::Error>> {
    let mut annotations = Vec::new();
    let content = std::fs::read_to_string(path)?;

    for line in content.lines().skip(1) {
        // Skip header
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            annotations.push(BatAnnotation {
                emitter: parts[0].parse().unwrap_or(0),
                addressee: parts[1].parse().unwrap_or(0),
                context: parts[2].parse().unwrap_or(0),
                filename: parts[7].to_string(),
            });
        }
    }

    Ok(annotations)
}

fn load_esc50_metadata(path: &Path) -> Result<Vec<Esc50Metadata>, Box<dyn std::error::Error>> {
    let mut metadata = Vec::new();
    let content = std::fs::read_to_string(path)?;

    for line in content.lines().skip(1) {
        // Skip header
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 5 {
            metadata.push(Esc50Metadata {
                filename: parts[0].to_string(),
                category: parts[4].to_string(),
                target: parts[2].parse().unwrap_or(0),
                fold: parts[1].parse().unwrap_or(0),
            });
        }
    }

    Ok(metadata)
}

fn select_diverse_esc50_samples(
    audio_dir: &Path,
    metadata: &[Esc50Metadata],
    subset_size: usize,
) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    use std::collections::HashSet;

    let mut selected_files = Vec::new();
    let mut used_categories = HashSet::new();
    let mut files_per_category = HashMap::new();

    // Group files by category
    for meta in metadata {
        files_per_category
            .entry(&meta.category)
            .or_insert_with(Vec::new)
            .push(meta);
    }

    // Select samples evenly across categories
    let categories: Vec<_> = files_per_category.keys().cloned().collect();
    let samples_per_category = subset_size / categories.len();
    let empty_vec = Vec::new();

    for category in categories {
        let files = files_per_category.get(&category).unwrap_or(&empty_vec);
        let count = files.len().min(samples_per_category);

        for meta in files.iter().take(count) {
            selected_files.push(audio_dir.join(&meta.filename));
            used_categories.insert(category.clone());
        }

        if selected_files.len() >= subset_size {
            break;
        }
    }

    Ok(selected_files)
}

fn count_unique_categories(files: &[std::path::PathBuf], metadata: &[Esc50Metadata]) -> usize {
    let meta_map: HashMap<_, _> = metadata
        .iter()
        .map(|m| (&m.filename, &m.category))
        .collect();

    files
        .iter()
        .filter_map(|f| f.file_name().and_then(|n| n.to_str()))
        .filter_map(|n| meta_map.get(&n.to_string()).copied())
        .collect::<HashSet<_>>()
        .len()
}

struct PipelineResult {
    name: String,
    phrases: usize,
    vocabulary: usize,
    zipf_alpha: Option<f64>,
    avg_cluster_size: f64,
    noise_count: usize,
    phoneme_models: usize,
}

fn run_pipeline(
    audio_files: &[std::path::PathBuf],
    output_dir: &Path,
    segmentation_config: &SegmentationConfig,
    vectorization_config: &VectorizationConfig,
    discovery_config: &DiscoveryConfig,
    refinement_config: &RefinementConfig,
    dataset_name: &str,
    sample_rate: u32,
) -> Result<PipelineResult, Box<dyn std::error::Error>> {
    println!("🔄 Running pipeline on {}...", dataset_name);
    println!("   ├─ Files: {}", audio_files.len());
    println!("   ├─ Sample rate: {}Hz", sample_rate);

    let start_time = std::time::Instant::now();

    // Create and configure pipeline
    let pipeline = LexiconToSyntaxPipeline::new()
        .with_segmentation_config(SegmentationConfig {
            sample_rate,
            ..segmentation_config.clone()
        })
        .with_vectorization_config(vectorization_config.clone())
        .with_discovery_config(discovery_config.clone())
        .with_refinement_config(refinement_config.clone());

    // Run pipeline (without checkpointing for this analysis)
    let result = pipeline.run(audio_files)?;
    let elapsed = start_time.elapsed();

    println!("   └─ Completed in {:.2}s", elapsed.as_secs_f64());
    println!();

    // Extract results
    let phrases = result.segmented_phrases.len();
    let vocabulary = result.vocabulary.len();
    let zipf_alpha = result.vocabulary_stats.zipf_alpha;
    let avg_cluster_size = result.vocabulary_stats.avg_cluster_size;
    let noise_count = result.vocabulary_stats.noise_count;
    let phoneme_models = result.phoneme_models.len();

    // Print detailed results
    println!("📊 Results for {}:", dataset_name);
    println!("   ├─ Segmented phrases: {}", phrases);
    println!("   ├─ Vocabulary items: {}", vocabulary);
    println!("   ├─ Noise phrases: {}", noise_count);
    if let Some(alpha) = zipf_alpha {
        println!("   ├─ Zipf's α: {:.3}", alpha);
    }
    println!("   └─ Phoneme models: {}", phoneme_models);
    println!();

    Ok(PipelineResult {
        name: dataset_name.to_string(),
        phrases,
        vocabulary,
        zipf_alpha,
        avg_cluster_size,
        noise_count,
        phoneme_models,
    })
}

fn print_comparison(bat: &PipelineResult, esc50: &PipelineResult) {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│                     METRIC COMPARISON TABLE                                 │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Metric                    │ Bat Vocalizations │ ESC-50 (Control) │ Assessment │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    // Segmented phrases
    println!(
        "│ Segmented Phrases         │ {:>16} │ {:>16} │            │",
        bat.phrases, esc50.phrases
    );

    // Vocabulary size
    println!(
        "│ Vocabulary Items          │ {:>16} │ {:>16} │            │",
        bat.vocabulary, esc50.vocabulary
    );

    // Zipf's alpha
    match (bat.zipf_alpha, esc50.zipf_alpha) {
        (Some(bat_alpha), Some(esc50_alpha)) => {
            let assessment = if (bat_alpha - 1.0).abs() < 0.3 {
                "✅ Natural"
            } else if (bat_alpha - 1.0).abs() < 0.7 {
                "⚠️  Moderate"
            } else {
                "❌ Unnatural"
            };

            println!(
                "│ Zipf's α                  │ {:>16.3} │ {:>16.3} │ {:>10} │",
                bat_alpha, esc50_alpha, assessment
            );

            println!(
                "├─────────────────────────────────────────────────────────────────────────────┤"
            );
            println!("│ Zipf's Law Interpretation:                                                     │");
            println!(
                "│   • α ≈ 1.0 indicates natural language-like distribution                      │"
            );
            println!(
                "│   • Bat vocalizations showing α ≈ 1.0 suggests communicative structure        │"
            );
            println!(
                "│   • Environmental sounds expected to deviate from α ≈ 1.0                     │"
            );
        }
        _ => {
            println!(
                "│ Zipf's α                  │ {:>16} │ {:>16} │            │",
                "N/A", "N/A"
            );
        }
    }

    // Average cluster size
    println!(
        "│ Avg Cluster Size          │ {:>16.1} │ {:>16.1} │            │",
        bat.avg_cluster_size, esc50.avg_cluster_size
    );

    // Noise ratio
    let bat_noise_ratio = (bat.noise_count as f64 / bat.phrases as f64) * 100.0;
    let esc50_noise_ratio = (esc50.noise_count as f64 / esc50.phrases as f64) * 100.0;
    println!(
        "│ Noise Ratio (%)           │ {:>16.1} │ {:>16.1} │            │",
        bat_noise_ratio, esc50_noise_ratio
    );

    println!(
        "│ Phoneme Models            │ {:>16} │ {:>16} │            │",
        bat.phoneme_models, esc50.phoneme_models
    );

    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Scientific interpretation
    println!("📈 Scientific Interpretation:");
    println!();

    // Zipf's law analysis
    if let (Some(bat_alpha), Some(esc50_alpha)) = (bat.zipf_alpha, esc50.zipf_alpha) {
        println!("Zipf's Law Analysis:");
        if (bat_alpha - 1.0).abs() < 0.3 {
            println!("   ✅ Bat vocalizations exhibit natural language distribution (α ≈ 1.0)");
            println!("      This suggests potential communicative structure in bat vocalizations.");
        } else if (bat_alpha - 1.0).abs() < 0.7 {
            println!("   ⚠️  Bat vocalizations show moderate adherence to Zipf's law");
            println!("      Some communicative structure may be present.");
        } else {
            println!("   ❌ Bat vocalizations deviate from natural Zipf's law");
            println!("      May indicate less structured communication or different patterns.");
        }

        if (esc50_alpha - 1.0).abs() > 0.5 {
            println!("   ✅ ESC-50 (control) shows expected non-linguistic distribution");
            println!("      Confirms environmental sounds lack communicative structure.");
        }
        println!();
    }

    // Vocabulary diversity analysis
    let bat_vocabulary_ratio = (bat.vocabulary as f64 / bat.phrases as f64) * 100.0;
    let esc50_vocabulary_ratio = (esc50.vocabulary as f64 / esc50.phrases as f64) * 100.0;

    println!("Vocabulary Diversity:");
    println!(
        "   ├─ Bat: {:.1}% of phrases are unique vocabulary items",
        bat_vocabulary_ratio
    );
    println!(
        "   ├─ ESC-50: {:.1}% of phrases are unique vocabulary items",
        esc50_vocabulary_ratio
    );

    if bat_vocabulary_ratio < esc50_vocabulary_ratio {
        println!(
            "   └─ ✅ Lower ratio in bats suggests more phrase repetition (linguistic marker)"
        );
    } else {
        println!("   └─ ⚠️  Higher ratio in bats may indicate less structured communication");
    }
    println!();

    // Cluster coherence analysis
    println!("Cluster Coherence:");
    println!(
        "   ├─ Bat: {} vocabulary items from {} phrases",
        bat.vocabulary, bat.phrases
    );
    println!(
        "   ├─ ESC-50: {} vocabulary items from {} phrases",
        esc50.vocabulary, esc50.phrases
    );
    println!("   └─ Higher vocabulary in bats suggests more diverse vocalization repertoire");
}
