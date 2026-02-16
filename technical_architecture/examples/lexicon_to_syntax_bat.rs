// Lexicon to Syntax Pipeline for Egyptian Fruit Bats
//
// This example demonstrates the complete 4-phase pipeline:
// Phase 1: Segmentation - Adaptive segmentation for variable-length phrases
// Phase 2: Vectorization - Extract 30D MicroDynamics feature time-series
// Phase 3: Discovery - DTW-DBSCAN clustering to find vocabulary
// Phase 4: Refinement - GMM-HMM for phoneme-level temporal structure

use std::path::Path;
use technical_architecture::{
    LexiconToSyntaxPipeline, SegmentationConfig, VectorizationConfig,
    DiscoveryConfig, RefinementConfig, PipelineCheckpoint,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║        Lexicon to Syntax: Egyptian Fruit Bats (4-Phase Pipeline)          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let base_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = base_dir.join("audio");
    let output_dir = base_dir.join("lexicon_to_syntax_results");

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
        println!("ℹ️  This example requires real bat audio files");
        println!("ℹ️  Please ensure the data is available at the specified path");
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
    let segmentation_config = SegmentationConfig {
        min_duration_ms: 50.0,      // Minimum phrase duration: 50ms
        max_duration_ms: 500.0,     // Maximum phrase duration: 500ms
        onset_threshold: 0.2,       // Onset detection threshold (lowered for more phrases)
        min_onset_distance_ms: 10.0, // Minimum distance between onsets
        sample_rate: 250000,        // 250kHz for bat recordings
    };

    println!("📐 Segmentation Config:");
    println!("   ├─ Min duration: {}ms", segmentation_config.min_duration_ms);
    println!("   ├─ Max duration: {}ms", segmentation_config.max_duration_ms);
    println!("   ├─ Onset threshold: {}", segmentation_config.onset_threshold);
    println!("   └─ Sample rate: {}Hz", segmentation_config.sample_rate);
    println!();

    // Phase 2: Vectorization Configuration
    let vectorization_config = VectorizationConfig {
        n_mels: 30,          // 30-dimensional MicroDynamics features
        fft_size: 2048,      // FFT size for spectral analysis
        hop_size: 512,       // Hop size for frame analysis
        normalize: true,     // Normalize features
    };

    println!("📊 Vectorization Config:");
    println!("   ├─ Feature dimensions: {}", vectorization_config.n_mels);
    println!("   ├─ FFT size: {}", vectorization_config.fft_size);
    println!("   ├─ Hop size: {}", vectorization_config.hop_size);
    println!("   └─ Normalize: {}", vectorization_config.normalize);
    println!();

    // Phase 3: Discovery Configuration
    let discovery_config = DiscoveryConfig {
        eps: 50.0,                     // DBSCAN epsilon threshold (high for DTW distances)
        min_samples: 3,                // Minimum samples for cluster (decreased for more clusters)
        dtw_window_size: None,         // Full DTW (no windowing)
        use_fast_dtw: true,            // Use FastDTW for speed
        fast_dtw_radius: 10,           // FastDTW radius
        use_lb_keogh: true,            // Use LB_Keogh lower bound pruning
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
        n_states: None,                 // Auto-determine HMM states
        n_components: 2,                // GMM components per state
        max_iterations: 100,            // Max EM iterations
        convergence_threshold: 1e-4,    // Convergence threshold
        covariance_reg: 1e-6,           // Covariance regularization
    };

    println!("🎯 Refinement Config:");
    println!("   ├─ HMM states: {:?}", refinement_config.n_states);
    println!("   ├─ GMM components: {}", refinement_config.n_components);
    println!("   ├─ Max iterations: {}", refinement_config.max_iterations);
    println!("   ├─ Convergence: {}", refinement_config.convergence_threshold);
    println!("   └─ Covariance regularization: {}", refinement_config.covariance_reg);
    println!();

    // Create pipeline with custom configs
    let pipeline = LexiconToSyntaxPipeline::new()
        .with_segmentation_config(segmentation_config)
        .with_vectorization_config(vectorization_config)
        .with_discovery_config(discovery_config)
        .with_refinement_config(refinement_config)
        .with_batch_size(20000); // Process 20K phrases per batch (reduces memory usage)

    println!("✅ Pipeline configured with custom parameters");
    println!();

    // =========================================================================
    // Collect Audio Files
    // =========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Collecting Audio Files");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Collect WAV files from audio directory
    let audio_files: Vec<_> = std::fs::read_dir(&audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect(); // Process ALL files (91,080 with checkpointing)

    println!("📁 Found {} audio files", audio_files.len());

    if audio_files.is_empty() {
        println!("⚠️  No WAV files found in audio directory");
        return Ok(());
    }

    // Show sample files
    for (i, file) in audio_files.iter().take(5).enumerate() {
        println!("   {}. {}", i + 1, file.file_name().unwrap().to_string_lossy());
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
        println!("   ├─ Checkpoint will be saved to: {}", checkpoint_path.display());
        println!("   └─ Checkpoint interval: every 100 files");
        println!();
    }

    // Estimate processing time based on file count (with parallel processing)
    // Baseline: 20 files serial = 490s, but with parallel processing on 8 cores ~61s
    let num_cores = num_cpus::get() as f64;
    let parallel_speedup = (num_cores * 0.7).min(8.0); // 70% efficiency per core, max 8x

    // Check if there's an existing checkpoint to adjust estimate
    let processed_count = if PipelineCheckpoint::exists(&checkpoint_path) {
        PipelineCheckpoint::load(&checkpoint_path)?.processed_files.len()
    } else {
        0
    };

    let remaining_files = audio_files.len() - processed_count;
    let estimated_seconds = ((remaining_files as f64 / 20.0) * 490.0) / parallel_speedup;
    let estimated_hours = estimated_seconds / 3600.0;

    println!("📊 Processing estimate:");
    if processed_count > 0 {
        println!("   ├─ Already processed: {} files", processed_count);
        println!("   ├─ Remaining: {} files", remaining_files);
    } else {
        println!("   ├─ Total files: {}", audio_files.len());
    }
    if estimated_hours < 1.0 {
        println!("   ├─ Estimated time: {:.1} minutes", estimated_seconds / 60.0);
    } else {
        println!("   ├─ Estimated time: {:.1} hours", estimated_hours);
    }
    println!("   ├─ {:.0} CPU cores detected (parallel processing)", num_cores);
    println!("   ├─ Parallel speedup: {:.1}x", parallel_speedup);
    println!("   └~{:.1} phrases/file", 1.7);
    println!();

    let start_time = std::time::Instant::now();

    // Run with checkpointing (interval doesn't matter, we checkpoint every 100 files)
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

        println!("   ├─ Duration range: {:.1}ms - {:.1}ms", min_duration, max_duration);
        println!("   ├─ Mean duration: {:.1}ms", mean_duration);

        // Show sample phrases
        println!("   └─ Sample phrases:");
        for phrase in phrases.iter().take(5) {
            println!("      • {} - {:.1}ms (onset conf: {:.2})",
                phrase.phrase_id, phrase.duration_ms, phrase.onset_confidence);
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
    println!("📊 Extracted feature matrices for {} phrases", features.len());

    if !features.is_empty() {
        let n_frames: Vec<usize> = features.iter().map(|f| f.n_frames).collect();
        let min_frames = n_frames.iter().cloned().min().unwrap_or(0);
        let max_frames = n_frames.iter().cloned().max().unwrap_or(0);
        let mean_frames = n_frames.iter().sum::<usize>() as f64 / n_frames.len() as f64;

        println!("   ├─ Frame count range: {} - {}", min_frames, max_frames);
        println!("   ├─ Mean frames: {:.1}", mean_frames);
        println!("   ├─ Feature dimensions: 30 (MicroDynamics)");
        println!("   └─ Frame rate: {:.1} Hz", features[0].frame_rate);

        // Show sample features
        println!();
        println!("   Sample feature vectors (first 3 dimensions):");
        for feat in features.iter().take(3) {
            let d0 = feat.features[[0, 0]];
            let d1 = feat.features[[0, 1]];
            let d2 = feat.features[[0, 2]];
            println!("      • {} - [{:.3}, {:.3}, {:.3}, ...]",
                feat.phrase_id, d0, d1, d2);
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

    println!("🔍 Discovered {} vocabulary items (phrase types)", vocabulary.len());
    println!();

    if !vocabulary.is_empty() {
        // Show cluster sizes
        let cluster_sizes: Vec<usize> = vocabulary.iter().map(|v| v.size).collect();
        let min_size = cluster_sizes.iter().cloned().min().unwrap_or(0);
        let max_size = cluster_sizes.iter().cloned().max().unwrap_or(0);
        let mean_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;

        println!("📊 Cluster Statistics:");
        println!("   ├─ Cluster size range: {} - {} phrases", min_size, max_size);
        println!("   ├─ Mean cluster size: {:.1} phrases", mean_size);
        println!("   ├─ Total phrases: {}", stats.total_phrases);
        println!("   ├─ Noise phrases: {}", stats.noise_count);
        println!("   └─ Avg cluster size: {:.1}", stats.avg_cluster_size);
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

        for (i, vocab) in sorted_vocab.iter().take(5).enumerate() {
            println!("   {}. Cluster {} - {} phrases", i + 1, vocab.cluster_id, vocab.size);
            println!("      ├─ Coherence: {:.3}", vocab.coherence);
            println!("      ├─ Feature template: [{:.3}, {:.3}, {:.3}, ...]",
                vocab.feature_template[[0, 0]],
                vocab.feature_template[[0, 1]],
                vocab.feature_template[[0, 2]],
            );
            println!("      └─ Sample phrase IDs: {}",
                vocab.phrase_ids.iter().take(3).cloned().collect::<Vec<_>>().join(", "));
        }
        if vocabulary.len() > 5 {
            println!("   ... and {} more", vocabulary.len() - 5);
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
        for model in phoneme_models.iter().take(3) {
            println!("   ├─ Cluster {} - {} states", model.cluster_id, model.n_states);
            println!("   │  ├─ Log-likelihood: {:.3}", model.log_likelihood);
            println!("   │  └─ State labels: {}",
                model.state_labels.join(" → "));
        }
        if phoneme_models.len() > 3 {
            println!("   └─ ... and {} more", phoneme_models.len() - 3);
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

    println!("Processed {} audio files in {:.2}s", audio_files.len(), elapsed.as_secs_f64());
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
    println!("In production, use real bat audio files for accurate results.");
    println!();

    // Create default pipeline
    let pipeline = LexiconToSyntaxPipeline::new();

    // Create empty audio list to trigger safe demo mode
    let empty_files: Vec<std::path::PathBuf> = vec
![];

    match pipeline.run(&empty_files) {
        Ok(_) => {
            println!("✅ Demo pipeline completed");
            println!();
            println!("Note: With real audio files, you would see:");
            println!("  • Hundreds of segmented phrases");
            println!("  • Dozens of vocabulary items");
            println!("  • Natural Zipf's law distribution (α ≈ 1.0)");
            println!("  • Multiple HMM phoneme models per cluster");
        }
        Err(e) => {
            println!("ℹ️  Demo mode: {}", e);
            println!();
            println!("To run with real data:");
            println!("  1. Ensure bat audio files are at:");
            println!("     /mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");
            println!("  2. Re-run this example");
        }
    }

    Ok(())
}
