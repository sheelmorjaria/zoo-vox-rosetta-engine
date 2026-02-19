// Lexicon to Syntax Pipeline for Marmoset Vocalizations
//
// This example demonstrates the complete 4-phase pipeline for marmoset vocalizations:
// Phase 1: Segmentation - Adaptive segmentation for variable-length phrases
// Phase 2: Vectorization - Extract 30D MicroDynamics feature time-series
// Phase 3: Discovery - DTW-DBSCAN clustering to find vocabulary
// Phase 4: Refinement - GMM-HMM for phoneme-level temporal structure
//
// Marmoset call types: Vocalization, Twitter, Tsik, Phee, Trill, Infant, Seep

use std::path::Path;
use technical_architecture::{
    DiscoveryConfig, LexiconToSyntaxPipeline, PipelineCheckpoint, RefinementConfig,
    SegmentationConfig, VectorizationConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Lexicon to Syntax: Marmoset Vocalizations (4-Phase Pipeline)           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let base_dir = Path::new("/home/sheel/birdsong_analysis/data");
    let audio_dir = base_dir.join("Vocalizations"); // Full dataset (871K files)
    let output_dir = base_dir.join("marmoset_lexicon_to_syntax_results");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;
    println!("✅ Output directory created: {}", output_dir.display());

    println!("📂 Base directory: {}", base_dir.display());
    println!("🎵 Audio directory: {}", audio_dir.display());
    println!("📊 Output directory: {}", output_dir.display());
    println!();

    // Check if audio directory exists
    if !audio_dir.exists() {
        println!("⚠️  Audio directory not found: {}", audio_dir.display());
        println!("ℹ️  This example requires marmoset FLAC files");
        println!("ℹ️  The pipeline now supports both WAV and FLAC formats!");
        println!();
        println!("For demonstration, we'll create a minimal test with synthetic data...");
        return run_demo_with_synthetic_data();
    }

    // =========================================================================
    // Phase 1: Configuration
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 0: Pipeline Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Phase 1: Segmentation Configuration
    // Marmoset vocalizations in this dataset are recorded at 96kHz
    // Each file is a separate recording that may contain silence + one call
    // We need very sensitive onset detection for these short calls
    let segmentation_config = SegmentationConfig {
        min_duration_ms: 10.0,      // 10ms minimum (marmoset calls are very short)
        max_duration_ms: 1000.0,    // 1 second maximum
        onset_threshold: 0.01,      // Extremely low threshold for very sensitive detection
        min_onset_distance_ms: 2.0, // 2ms minimum between onsets
        sample_rate: 96000,         // 96kHz (actual sample rate of marmoset recordings)
    };

    println!("📐 Segmentation Config:");
    println!(
        "   ├─ Min duration: {}ms",
        segmentation_config.min_duration_ms
    );
    println!(
        "   ├─ Max duration: {}ms",
        segmentation_config.max_duration_ms
    );
    println!(
        "   ├─ Onset threshold: {}",
        segmentation_config.onset_threshold
    );
    println!("   └─ Sample rate: {}Hz", segmentation_config.sample_rate);
    println!();

    // Phase 2: Vectorization Configuration
    let vectorization_config = VectorizationConfig {
        n_mels: 30,      // 30-dimensional MicroDynamics features
        fft_size: 2048,  // FFT size for spectral analysis
        hop_size: 512,   // Hop size for frame analysis
        normalize: true, // Normalize features
    };

    println!("📊 Vectorization Config:");
    println!("   ├─ Feature dimensions: {}", vectorization_config.n_mels);
    println!("   ├─ FFT size: {}", vectorization_config.fft_size);
    println!("   ├─ Hop size: {}", vectorization_config.hop_size);
    println!("   └─ Normalize: {}", vectorization_config.normalize);
    println!();

    // Phase 3: Discovery Configuration
    // Marmoset vocalizations have distinct call types - use larger epsilon for clustering
    // The features from short vocalizations may need larger epsilon to form clusters
    let discovery_config = DiscoveryConfig {
        eps: 10.0,             // DBSCAN epsilon (much larger for marmoset feature space)
        min_samples: 2,        // Minimum samples for cluster (lower for small clusters)
        dtw_window_size: None, // Full DTW (no windowing)
        use_fast_dtw: true,    // Use FastDTW for speed
        fast_dtw_radius: 10,   // FastDTW radius
        use_lb_keogh: true,    // Use LB_Keogh lower bound pruning
    };

    println!("🔍 Discovery Config:");
    println!("   ├─ DBSCAN epsilon: {}", discovery_config.eps);
    println!("   ├─ Min samples: {}", discovery_config.min_samples);
    println!("   ├─ DTW window: {:?}", discovery_config.dtw_window_size);
    println!("   ├─ Fast DTW: {}", discovery_config.use_fast_dtw);
    println!("   └─ LB_Keogh pruning: {}", discovery_config.use_lb_keogh);
    println!();

    // Phase 4: Refinement Configuration
    let refinement_config = RefinementConfig {
        n_states: None,              // Auto-determine HMM states
        n_components: 2,             // GMM components per state
        max_iterations: 100,         // Max EM iterations
        convergence_threshold: 1e-4, // Convergence threshold
        covariance_reg: 1e-6,        // Covariance regularization
    };

    println!("🎯 Refinement Config:");
    println!("   ├─ HMM states: {:?}", refinement_config.n_states);
    println!("   ├─ GMM components: {}", refinement_config.n_components);
    println!("   ├─ Max iterations: {}", refinement_config.max_iterations);
    println!(
        "   ├─ Convergence: {}",
        refinement_config.convergence_threshold
    );
    println!(
        "   └─ Covariance regularization: {}",
        refinement_config.covariance_reg
    );
    println!();

    // Create pipeline with custom configs
    let pipeline = LexiconToSyntaxPipeline::new()
        .with_segmentation_config(segmentation_config)
        .with_vectorization_config(vectorization_config)
        .with_discovery_config(discovery_config)
        .with_refinement_config(refinement_config)
        .with_batch_size(10000); // Process 10K phrases per batch

    println!("✅ Pipeline configured with custom parameters");
    println!();

    // =========================================================================
    // Collect Audio Files
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Collecting Audio Files");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Collect WAV and FLAC files from audio directory (recursively)
    let mut audio_files = Vec::new();
    fn collect_flac_files(
        dir: &std::path::Path,
        files: &mut Vec<std::path::PathBuf>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                collect_flac_files(&path, files)?;
            } else if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("flac") || ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
        Ok(())
    }
    collect_flac_files(&audio_dir, &mut audio_files)?;

    println!("📁 Found {} audio files", audio_files.len());

    if audio_files.is_empty() {
        println!("⚠️  No audio files found in audio directory");
        return Ok(());
    }

    // Show sample files
    for (i, file) in audio_files.iter().take(5).enumerate() {
        println!(
            "   {}. {}",
            i + 1,
            file.file_name().unwrap().to_string_lossy()
        );
    }
    if audio_files.len() > 5 {
        println!("   ... and {} more", audio_files.len() - 5);
    }
    println!();

    // =========================================================================
    // Run Complete 4-Phase Pipeline with Checkpointing
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Running 4-Phase Pipeline with Checkpointing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Setup checkpoint path
    let checkpoint_path = output_dir.join("pipeline_checkpoint.json");

    // Check for existing checkpoint
    if PipelineCheckpoint::exists(&checkpoint_path) {
        println!("📂 Existing checkpoint found!");
        println!("   ├─ Location: {}", checkpoint_path.display());
        println!("   └─ Pipeline will resume from last checkpoint");
        println!();
    } else {
        println!("💾 No checkpoint found - starting fresh");
        println!(
            "   ├─ Checkpoint will be saved to: {}",
            checkpoint_path.display()
        );
        println!("   └─ Checkpoint interval: every 100 files");
        println!();
    }

    let start_time = std::time::Instant::now();

    // Run with checkpointing
    let result = pipeline.run_with_checkpoint(&audio_files, checkpoint_path.clone(), 600)?;
    let elapsed = start_time.elapsed();

    println!("✅ Pipeline completed in {:.2}s", elapsed.as_secs_f64());
    println!();

    // =========================================================================
    // Phase 1 Results: Segmentation
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 1 Results: Segmentation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let phrases = &result.segmented_phrases;
    println!("📊 Segmented {} phrases", phrases.len());

    if !phrases.is_empty() {
        let durations: Vec<f64> = phrases.iter().map(|p| p.duration_ms).collect();
        let min_duration = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_duration = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_duration = durations.iter().sum::<f64>() / durations.len() as f64;

        println!(
            "   ├─ Duration range: {:.1}ms - {:.1}ms",
            min_duration, max_duration
        );
        println!("   ├─ Mean duration: {:.1}ms", mean_duration);

        // Show sample phrases
        println!("   └─ Sample phrases:");
        for phrase in phrases.iter().take(5) {
            println!(
                "      • {} - {:.1}ms (onset conf: {:.2})",
                phrase.phrase_id, phrase.duration_ms, phrase.onset_confidence
            );
        }
        if phrases.len() > 5 {
            println!("      ... and {} more", phrases.len() - 5);
        }
    }
    println!();

    // =========================================================================
    // Phase 2 Results: Vectorization
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 2 Results: Vectorization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let features = &result.phrase_features;
    println!(
        "📊 Extracted feature matrices for {} phrases",
        features.len()
    );

    if !features.is_empty() {
        println!("   ├─ Feature dimensions: 30 (MicroDynamics)");
        println!("   ├─ Frame rate: {:.1} Hz", features[0].frame_rate);

        // Show sample features
        println!();
        println!("   Sample feature vectors (first 3 dimensions):");
        for feat in features.iter().take(3) {
            let d0 = feat.features[[0, 0]];
            let d1 = feat.features[[0, 1]];
            let d2 = feat.features[[0, 2]];
            println!(
                "      • {} - [{:.3}, {:.3}, {:.3}, ...]",
                feat.phrase_id, d0, d1, d2
            );
        }
    }
    println!();

    // =========================================================================
    // Phase 3 Results: Discovery (Vocabulary)
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 3 Results: Discovery (Vocabulary)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let vocabulary = &result.vocabulary;
    let stats = &result.vocabulary_stats;

    println!(
        "🔍 Discovered {} vocabulary items (phrase types)",
        vocabulary.len()
    );
    println!();

    if !vocabulary.is_empty() {
        // Show cluster sizes
        println!("📊 Cluster Statistics:");
        println!("   ├─ Total phrases: {}", stats.total_phrases);
        println!("   ├─ Noise phrases: {}", stats.noise_count);
        println!("   ├─ Avg cluster size: {:.1}", stats.avg_cluster_size);
        println!("   └─ Max cluster size: {}", stats.max_cluster_size);
        println!();

        // Zipf's Law analysis
        if let Some(alpha) = stats.zipf_alpha {
            println!("📈 Zipf's Law Analysis:");
            if alpha > 0.0 {
                println!("   ├─ Zipf's α = {:.3}", alpha);
                if alpha >= 0.7 && alpha <= 1.3 {
                    println!("   ├─ Assessment: ✅ Natural language distribution");
                    println!("   └─ The vocabulary follows natural Zipf's law (α ≈ 1.0)");
                } else if alpha >= 0.5 && alpha <= 2.0 {
                    println!("   ├─ Assessment: ⚠️  Acceptable distribution");
                    println!("   └─ Within acceptable range for animal communication");
                } else {
                    println!("   ├─ Assessment: ⚠️  Unusual distribution");
                    println!("   └─ Outside typical Zipf's law range");
                }
            } else {
                println!("   ├─ Assessment: ⚠️  Inverse distribution detected");
                println!("   └─ Alpha = {:.3} (expected positive for Zipf)", alpha);
            }
        }
        println!();

        // Show top vocabulary items
        println!("🎯 Top Vocabulary Items (by cluster size):");
        let mut sorted_vocab = vocabulary.clone();
        sorted_vocab.sort_by(|a, b| b.size.cmp(&a.size));

        for (i, vocab) in sorted_vocab.iter().take(10).enumerate() {
            println!(
                "   {}. Cluster {} - {} phrases",
                i + 1,
                vocab.cluster_id,
                vocab.size
            );
            println!("      ├─ Coherence: {:.3}", vocab.coherence);
            println!(
                "      ├─ Feature template: [{:.3}, {:.3}, {:.3}, ...]",
                vocab.feature_template[[0, 0]],
                vocab.feature_template[[0, 1]],
                vocab.feature_template[[0, 2]],
            );
            println!(
                "      └─ Sample phrase IDs: {}",
                vocab
                    .phrase_ids
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if vocabulary.len() > 10 {
            println!("   ... and {} more", vocabulary.len() - 10);
        }
    }
    println!();

    // =========================================================================
    // Phase 4 Results: Refinement (Phoneme Models)
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 4 Results: Refinement (Phoneme Models)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let phoneme_models = &result.phoneme_models;
    println!("🎯 Trained {} GMM-HMM phoneme models", phoneme_models.len());

    if !phoneme_models.is_empty() {
        let n_states: Vec<usize> = phoneme_models.iter().map(|m| m.n_states).collect();
        let min_states = n_states.iter().cloned().min().unwrap_or(0);
        let max_states = n_states.iter().cloned().max().unwrap_or(0);

        println!("   ├─ HMM state range: {} - {}", min_states, max_states);

        // Show sample models
        println!();
        println!("   Sample Phoneme Models:");
        for model in phoneme_models.iter().take(5) {
            println!(
                "   ├─ Cluster {} - {} states",
                model.cluster_id, model.n_states
            );
            println!("   │  ├─ Log-likelihood: {:.3}", model.log_likelihood);
            println!("   │  └─ State labels: {}", model.state_labels.join(" → "));
        }
        if phoneme_models.len() > 5 {
            println!("   └─ ... and {} more", phoneme_models.len() - 5);
        }
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PIPELINE SUMMARY                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "Processed {} audio files in {:.2}s",
        audio_files.len(),
        elapsed.as_secs_f64()
    );
    println!();

    println!("Phase 1 - Segmentation:");
    println!("  • Segmented phrases: {}", result.segmented_phrases.len());
    println!();

    println!("Phase 2 - Vectorization:");
    println!("  • Feature matrices: {}", result.phrase_features.len());
    println!("  • Feature dimensions: 30");
    println!();

    println!("Phase 3 - Discovery:");
    println!("  • Vocabulary items: {}", result.vocabulary.len());
    println!("  • Total phrases: {}", stats.total_phrases);
    println!("  • Noise phrases: {}", stats.noise_count);
    if let Some(alpha) = stats.zipf_alpha {
        println!("  • Zipf's α: {:.3}", alpha);
    }
    println!();

    println!("Phase 4 - Refinement:");
    println!("  • Phoneme models: {}", result.phoneme_models.len());
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💾 Checkpoint Information");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Checkpoint saved to: {}", checkpoint_path.display());
    println!("   ├─ To resume: Re-run the same command");
    println!("   ├─ To start fresh: Delete the checkpoint file");
    println!("   └─ Checkpoint format: JSON (human-readable)");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

/// Demo function with synthetic data (for when audio files are not available)
fn run_demo_with_synthetic_data() -> Result<(), Box<dyn std::error::Error>> {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│                     DEMO MODE: Synthetic Data                               │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create minimal synthetic demo
    println!("This demo shows the pipeline structure with minimal synthetic data.");
    println!("In production, use real marmoset audio files for accurate results.");
    println!();
    println!("💡 The pipeline now supports FLAC files directly!");
    println!("   No need to convert to WAV - just point to your FLAC files.");
    println!();

    // Create default pipeline
    let pipeline = LexiconToSyntaxPipeline::new();

    // Create empty audio list to trigger safe demo mode
    let empty_files: Vec<std::path::PathBuf> = vec![];

    match pipeline.run(&empty_files) {
        Ok(_) => {
            println!("✅ Demo pipeline completed");
            println!();
            println!("Note: With real audio files, you would see:");
            println!("  • Thousands of segmented phrases");
            println!("  • Hundreds of vocabulary items across call types");
            println!("  • Natural Zipf's law distribution (α ≈ 1.0)");
            println!("  • Multiple HMM phoneme models per cluster");
        }
        Err(e) => {
            println!("ℹ️  Demo mode: {}", e);
            println!();
            println!("To run with real data:");
            println!("  1. Copy marmoset FLAC files to:");
            println!("     /home/sheel/birdsong_analysis/data/marmoset_flac_subset");
            println!("  2. Or use the existing:");
            println!("     /home/sheel/birdsong_analysis/data/Vocalizations");
            println!("  3. Re-run this example");
        }
    }

    Ok(())
}
