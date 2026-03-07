// Vocabulary Mapping for Egyptian Fruit Bats
//
// This example demonstrates the complete pipeline:
// 1. Load clustered phrases from parallel extraction
// 2. Map vocabulary to context annotations
// 3. Extract audio segments for synthesis
// 4. Generate synthesis assets

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::path::Path;
use technical_architecture::{
    AnnotationDataset, AudioSegmenter, ConcatenativeParams, GrainEnvelopeType, GranularSynthesisParams,
    MetadataDrivenParams, SynthesisPipeline, VocabularyMapper, VocalizationContext,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║           Vocabulary Mapping: Egyptian Fruit Bats                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let base_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = base_dir.join("audio");
    let annotations_path = base_dir.join("Annotations.csv");
    let output_dir = base_dir.join("extraction_results_optimized");

    println!("📂 Base directory: {}", base_dir.display());
    println!("🎵 Audio directory: {}", audio_dir.display());
    println!("📝 Annotations: {}", annotations_path.display());
    println!("📊 Output directory: {}", output_dir.display());
    println!();

    // Step 1: Load annotations
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Loading Context Annotations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let annotations = load_bat_annotations(&annotations_path)?;
    println!("   ✅ Loaded {} annotations", annotations.annotations.len());
    println!();

    // Step 2: Create vocabulary mapper
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Creating Vocabulary Mapper");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let sample_rate = 250000; // 250kHz for bat recordings
    let mut mapper = VocabularyMapper::new(annotations, sample_rate);
    println!("   ✅ Vocabulary mapper created (sample_rate = {}Hz)", sample_rate);
    println!();

    // Step 3: Load clustered phrases from JSON
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Loading Clustered Phrases");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let phrases_json = output_dir.join("phrases_metadata.json");
    if !phrases_json.exists() {
        println!("   ⚠️  Phrases metadata not found at: {}", phrases_json.display());
        println!("   Please run full_pipeline_bat example first to generate clustered phrases");
        println!("   Command: cargo run --example full_pipeline_bat --release");
        return Ok(());
    }

    // For this demo, we'll create sample vocabulary mapping
    // In production, you'd load from the clustered phrases JSON
    println!("   ℹ️  Note: Creating sample vocabulary mapping for demonstration");
    println!("   ℹ️  In production, load from phrases_metadata.json");

    // Create sample cluster assignments
    // In production, these would come from your clustering results
    let cluster_labels = vec![0; 100]; // 100 files in cluster 0
    let file_paths: Vec<String> = (0..100).map(|i| format!("sentence_{:05}.wav", i)).collect();
    let time_ranges: Vec<(f64, f64)> = (0..100).map(|i| (i as f64 * 0.5, i as f64 * 0.5 + 0.3)).collect();

    // Create sample feature series (30D features)
    use ndarray::Array2;
    let feature_series: Vec<Array2<f64>> = (0..100)
        .map(|_| {
            // Create 10 time steps x 30 dimensions
            Array2::zeros((10, 30))
        })
        .collect();

    println!("   📊 Mapping {} phrases to vocabulary", file_paths.len());

    // Map vocabulary
    mapper.map_vocabulary(&cluster_labels, &file_paths, &time_ranges, &feature_series)?;
    println!("   ✅ Vocabulary mapped");
    println!();

    // Step 4: Display vocabulary statistics
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 4: Vocabulary Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let stats = mapper.get_statistics();
    println!("   📊 Total vocabulary items: {}", stats.total_vocabulary_items);
    println!("   📊 Total occurrences: {}", stats.total_occurrences);
    println!("   📊 Unique contexts: {}", stats.unique_contexts);
    println!();

    // Display vocabulary items
    for vocab_id in mapper.vocabulary_ids() {
        if let Some(vocab) = mapper.get_vocabulary(&vocab_id) {
            println!("   🎯 Vocabulary: {}", vocab_id);
            println!("      ├─ Cluster: {}", vocab.cluster_id);
            println!("      ├─ Occurrences: {}", vocab.occurrences.len());
            println!(
                "      ├─ Duration: {:.1}ms - {:.1}ms (mean: {:.1}ms)",
                vocab.duration_stats.min_ms, vocab.duration_stats.max_ms, vocab.duration_stats.mean_ms
            );
            println!("      ├─ Feature templates: {}", vocab.feature_templates.len());

            // Show contexts
            let mut contexts = std::collections::HashSet::new();
            for occ in &vocab.occurrences {
                if let Some(ctx) = &occ.context.behavioral_context {
                    contexts.insert(ctx.as_str());
                }
            }
            if !contexts.is_empty() {
                println!(
                    "      └─ Contexts: {}",
                    contexts.iter().cloned().collect::<Vec<_>>().join(", ")
                );
            } else {
                println!("      └─ Contexts: None");
            }
            println!();
        }
    }

    // Step 5: Create audio segmenter
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 5: Creating Audio Segmenter");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let synthesis_output_dir = base_dir.join("synthesis_assets");
    let segmenter = AudioSegmenter::new(&audio_dir, &synthesis_output_dir, sample_rate);
    println!("   ✅ Audio segmenter created");
    println!("   📂 Output directory: {}", synthesis_output_dir.display());
    println!();

    // Step 6: Create synthesis pipeline
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 6: Creating Synthesis Pipeline");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let pipeline = SynthesisPipeline::new(mapper.clone(), segmenter, &synthesis_output_dir);
    println!("   ✅ Synthesis pipeline created");
    println!();

    // Step 7: Generate synthesis assets
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 7: Generating Synthesis Assets");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    match pipeline.generate_synthesis_assets() {
        Ok(assets) => {
            println!("   ✅ Synthesis assets generated:");
            println!("      ├─ Metadata file: {}", assets.metadata_path.display());
            println!("      ├─ Concatenative assets: {}", assets.concatenative_assets.len());
            println!("      ├─ Granular assets: {}", assets.granular_assets.len());
            println!("      └─ Metadata assets: {}", assets.metadata_assets.len());

            // Export vocabulary statistics
            let stats_json = synthesis_output_dir.join("vocabulary_stats.json");
            mapper.export_json(&stats_json)?;
            println!("   ✅ Vocabulary statistics exported: {}", stats_json.display());
        }
        Err(e) => {
            println!("   ⚠️  Asset generation failed: {}", e);
            println!("   ℹ️  This is expected if audio files don't exist at the specified paths");
        }
    }
    println!();

    // Step 8: Test synthesis techniques
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 8: Testing Synthesis Techniques");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Test metadata-driven synthesis
    if let Some(vocab_id) = mapper.vocabulary_ids().first() {
        println!("   🎵 Testing metadata-driven synthesis for: {}", vocab_id);

        let params = MetadataDrivenParams {
            target_duration: 0.3, // 300ms
            f0_contour: vec![15000.0, 18000.0, 16000.0],
            intensity: 0.7,
            spectral_centroid: 25000.0,
            spectral_bandwidth: 10000.0,
        };

        match pipeline.synthesize_metadata_driven(vocab_id, &params) {
            Ok(audio) => {
                println!("      ✅ Generated {} samples", audio.len());
            }
            Err(e) => {
                println!("      ⚠️  Failed: {}", e);
            }
        }
    }

    // Test granular synthesis
    if let Some(vocab_id) = mapper.vocabulary_ids().first() {
        println!("   🎵 Testing granular synthesis for: {}", vocab_id);

        let params = GranularSynthesisParams {
            grain_size_ms: 50.0,
            hop_size_ms: 25.0,
            envelope: GrainEnvelopeType::Hann,
            density: 0.5,
            time_stretch: 1.0,
            pitch_shift: 1.0,
        };

        match pipeline.synthesize_granular(vocab_id, &params, 1.0) {
            Ok(audio) => {
                println!("      ✅ Generated {} samples (1.0s)", audio.len());
            }
            Err(e) => {
                println!("      ⚠️  Failed: {}", e);
            }
        }
    }

    // Test concatenative synthesis
    if let Some(vocab_id) = mapper.vocabulary_ids().first() {
        println!("   🎵 Testing concatenative synthesis for: {}", vocab_id);

        let params = ConcatenativeParams {
            crossfade_ms: 10.0,
            normalize: true,
            min_segment_duration: 0.1,
            max_segment_duration: 0.5,
        };

        match pipeline.synthesize_concatenative(vocab_id, &params) {
            Ok(audio) => {
                println!("      ✅ Generated {} samples", audio.len());
            }
            Err(e) => {
                println!("      ⚠️  Failed: {}", e);
            }
        }
    }
    println!();

    // Final summary
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    VOCABULARY MAPPING COMPLETE                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Summary:");
    println!("  • Vocabulary items discovered: {}", stats.total_vocabulary_items);
    println!("  • Total occurrences: {}", stats.total_occurrences);
    println!("  • Unique behavioral contexts: {}", stats.unique_contexts);
    println!("  • Synthesis assets: {}/synthesis_assets", base_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. Review synthesized audio in synthesis_assets/concatenative/");
    println!("  2. Review metadata in synthesis_assets/metadata/");
    println!("  3. Review grains in synthesis_assets/granular/");
    println!("  4. Use synthesis_pipeline module for advanced synthesis techniques");
    println!();

    Ok(())
}

/// Load bat annotations from CSV
/// For Egyptian fruit bats, this includes:
/// - Emitter ID (which bat)
/// - Behavioral context (feeding, aggression, mating, etc.)
/// - Time of day
/// - Location
fn load_bat_annotations(path: &Path) -> Result<AnnotationDataset, Box<dyn std::error::Error>> {
    // For this example, create sample annotations
    // In production, load from actual CSV file

    if path.exists() {
        println!("   ℹ️  Loading annotations from: {}", path.display());
        // TODO: Implement actual CSV loading
        return Ok(AnnotationDataset { annotations: vec![] });
    }

    println!("   ℹ️  Creating sample annotations for demonstration");

    // Create sample annotations matching our file paths
    let mut annotations = Vec::new();

    for i in 0..100 {
        let file_name = format!("sentence_{:05}.wav", i);

        // Sample behavioral contexts
        let contexts = vec!["feeding", "aggression", "mating", "parental_care", "unknown"];
        let context = contexts[i % contexts.len()];

        // Sample emitters
        let emitters = vec!["bat_1", "bat_2", "bat_3", "bat_4"];
        let emitter = emitters[i % emitters.len()];

        annotations.push(VocalizationContext {
            file_path: file_name.clone(),
            start_time: i as f64 * 0.5,
            end_time: i as f64 * 0.5 + 0.3,
            emitter_id: Some(emitter.to_string()),
            addressee_id: None,
            behavioral_context: Some(context.to_string()),
            time_of_day: Some("night".to_string()),
            location: Some("colony".to_string()),
            social_context: Some("group".to_string()),
            environmental_conditions: Some("indoor".to_string()),
        });
    }

    Ok(AnnotationDataset { annotations })
}
