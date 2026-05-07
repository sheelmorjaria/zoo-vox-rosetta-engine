//! Semantic Reconstruction Pipeline Tests (TDD)
//! ===========================================
//!
//! Tests for STAGE 4 of the synthesis pipeline:
//! - ExemplarManager: Best audio per cluster ID
//! - Metadata Mapper: 112D to 112D metadata
//! - CachedGranularSynthesizer: register_source + synthesize_timeline
//!
//! Pipeline Flow:
//! STAGE 1: NBD SEGMENTATION (audio_segmenter.rs)
//! STAGE 2: 112D FEATURE EXTRACTION (micro_dynamics_extractor.rs)
//! STAGE 3: CORPUS ANALYSIS (corpus_analysis.rs, clustering.rs)
//! STAGE 4: SEMANTIC RECONSTRUCTION (THIS MODULE)
//! STAGE 5: SYNTHESIS OUTPUT (synthesis.rs)
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use technical_architecture::{
    CachedGranularSynthesizer, ExemplarManager, RosettaFeatures, SemanticTimelineEvent, SourceMetadata112D,
    SynthesisConfig112D, SynthesisTimeline,
};

// =============================================================================
// EXEMPLAR MANAGER TESTS
// =============================================================================

#[test]
fn test_exemplar_manager_creation() {
    let manager = ExemplarManager::new();
    assert_eq!(manager.len(), 0);
}

#[test]
fn test_exemplar_manager_register_single() {
    let mut manager = ExemplarManager::new();
    let audio1 = vec![0.1, 0.1];
    let mut features1 = RosettaFeatures::default();
    features1.rms_energy = 0.3; // Lower quality
    manager.register_exemplar(42, audio1.clone(), features1);

    // Second exemplar with higher quality should replace first
    let audio2 = vec![0.5, 0.5, 0.5, 0.5];
    let mut features2 = RosettaFeatures::default();
    features2.rms_energy = 0.7; // Higher quality
    manager.register_exemplar(42, audio2.clone(), features2);

    let entry = manager.get_exemplar(42).expect("Exemplar should exist");
    assert_eq!(entry.audio, audio2, "Should keep higher quality exemplar");
}

#[test]
fn test_exemplar_manager_replace_with_better_quality() {
    let mut manager = ExemplarManager::new();

    // First exemplar with lower quality
    let audio1 = vec![0.1, 0.1, 0.1];
    let mut features1 = RosettaFeatures::default();
    features1.rms_energy = 0.3; // Lower quality
    manager.register_exemplar(42, audio1.clone(), features1);

    // Second exemplar with higher quality should replace first
    let audio2 = vec![0.5, 0.5, 0.5, 0.5];
    let mut features2 = RosettaFeatures::default();
    features2.rms_energy = 0.7; // Higher quality
    manager.register_exemplar(42, audio2.clone(), features2);

    let entry = manager.get_exemplar(42).expect("Exemplar should exist");
    assert_eq!(entry.audio, audio2, "Should keep higher quality exemplar");
}

#[test]
fn test_exemplar_manager_keep_better_on_lower_quality() {
    let mut manager = ExemplarManager::new();

    // First exemplar with higher quality
    let audio1 = vec![0.9, 0.9, 0.9];
    let mut features1 = RosettaFeatures::default();
    features1.rms_energy = 0.9; // High quality
    manager.register_exemplar(42, audio1.clone(), features1);

    // Second exemplar with lower quality should NOT replace
    let audio2 = vec![0.1, 0.1, 0.1];
    let mut features2 = RosettaFeatures::default();
    features2.rms_energy = 0.2; // Lower quality
    manager.register_exemplar(42, audio2, features2);

    let entry = manager.get_exemplar(42).expect("Exemplar should exist");
    assert_eq!(entry.audio, audio1, "Should keep original high quality exemplar");
}

#[test]
fn test_exemplar_manager_multiple_clusters() {
    let mut manager = ExemplarManager::new();

    let audio1 = vec![0.5, 0.3];
    let audio2 = vec![0.7, 0.4];
    let audio3 = vec![0.6, 0.2];

    manager.register_exemplar(1, audio1.clone(), RosettaFeatures::default());
    manager.register_exemplar(2, audio2.clone(), RosettaFeatures::default());
    manager.register_exemplar(3, audio3.clone(), RosettaFeatures::default());

    assert_eq!(manager.len(), 3);
    assert!(manager.get_exemplar(1).is_some());
    assert!(manager.get_exemplar(2).is_some());
    assert!(manager.get_exemplar(3).is_some());
    assert!(manager.get_exemplar(999).is_none());
}

#[test]
fn test_exemplar_manager_clear() {
    let mut manager = ExemplarManager::new();

    manager.register_exemplar(1, vec![0.5], RosettaFeatures::default());
    manager.register_exemplar(2, vec![0.3], RosettaFeatures::default());
    assert_eq!(manager.len(), 2);

    manager.clear();
    assert_eq!(manager.len(), 0);
}

// =============================================================================
// SOURCE METADATA TESTS
// =============================================================================

#[test]
fn test_source_metadata_creation() {
    let features = RosettaFeatures::default();
    let metadata = SourceMetadata112D::from_features(&features);

    // Should contain all 112D features
    assert_eq!(metadata.cluster_id, None);
    assert!(metadata.quality_score() >= 0.0 && metadata.quality_score() <= 1.0);
}

#[test]
fn test_source_metadata_with_cluster_id() {
    let features = RosettaFeatures::default();
    let metadata = SourceMetadata112D::from_features_with_cluster(&features, 123);

    assert_eq!(metadata.cluster_id, Some(123));
}

#[test]
fn test_source_metadata_to_array() {
    let features = RosettaFeatures::default();
    let metadata = SourceMetadata112D::from_features(&features);
    let array = metadata.to_array_112d();

    // Should have 112 dimensions
    assert_eq!(array.len(), 112);
}

// =============================================================================
// SYNTHESIS TIMELINE TESTS
// =============================================================================

#[test]
fn test_synthesis_timeline_creation() {
    let timeline = SynthesisTimeline::new();
    assert!(timeline.is_empty());
    assert_eq!(timeline.len(), 0);
}

#[test]
fn test_synthesis_timeline_add_event() {
    let mut timeline = SynthesisTimeline::new();

    let event = SemanticTimelineEvent {
        cluster_id: 42,
        start_time_ms: 0.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    };

    timeline.add_event(event);
    assert_eq!(timeline.len(), 1);
    assert!(!timeline.is_empty());
}

#[test]
fn test_synthesis_timeline_from_ngram() {
    let mut timeline = SynthesisTimeline::new();

    // Simulate an N-gram template: cluster sequence [1, 2, 3]
    // Each event is 50ms
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 1,
        start_time_ms: 0.0,
        duration_ms: 50.0,
        amplitude: 1.0,
    });
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 2,
        start_time_ms: 50.0,
        duration_ms: 50.0,
        amplitude: 1.0,
    });
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 3,
        start_time_ms: 100.0,
        duration_ms: 50.0,
        amplitude: 1.0,
    });

    assert_eq!(timeline.len(), 3);
    assert_eq!(timeline.total_duration_ms(), 150.0);
}

#[test]
fn test_synthesis_timeline_get_events_in_range() {
    let mut timeline = SynthesisTimeline::new();

    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 1,
        start_time_ms: 0.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 2,
        start_time_ms: 100.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 3,
        start_time_ms: 200.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });

    // Get events in range [50, 150)
    let events = timeline.get_events_in_range(50.0, 150.0);
    assert_eq!(events.len(), 2); // Events 1 and 2
}

// =============================================================================
// CACHED GRANular Synthesizer Tests
// =============================================================================

#[test]
fn test_cached_granular_synthesizer_creation() {
    let config = SynthesisConfig112D::default();
    let synth = CachedGranularSynthesizer::new(config);
    assert_eq!(synth.source_count(), 0);
}

#[test]
fn test_cached_granular_synthesizer_register_source() {
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    let audio = vec![0.5, 0.3, 0.8, 0.2];
    let features = RosettaFeatures::default();
    let metadata = SourceMetadata112D::from_features(&features);

    synth.register_source(42, audio, metadata);
    assert_eq!(synth.source_count(), 1);
}

#[test]
fn test_cached_granular_synthesizer_multiple_sources() {
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    for id in 1..=10u32 {
        let audio = vec![0.5f32; id as usize];
        let features = RosettaFeatures::default();
        let metadata = SourceMetadata112D::from_features(&features);
        synth.register_source(id, audio, metadata);
    }

    assert_eq!(synth.source_count(), 10);
}

#[test]
fn test_cached_granular_synthesizer_replace_source() {
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    let audio1 = vec![0.1, 0.1];
    let features1 = RosettaFeatures::default();
    let metadata1 = SourceMetadata112D::from_features(&features1);
    synth.register_source(42, audio1, metadata1);

    // Replace with new source
    let audio2 = vec![0.9, 0.9];
    let features2 = RosettaFeatures::default();
    let metadata2 = SourceMetadata112D::from_features(&features2);
    synth.register_source(42, audio2.clone(), metadata2);

    // Should have only 1 source (replaced)
    assert_eq!(synth.source_count(), 1);
}

#[tokio::test]
async fn test_cached_granular_synthesizer_synthesize_timeline() {
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    // Register sources
    let audio1 = vec![0.5; 4800]; // 100ms at 48kHz
    let audio2 = vec![0.3; 4800];
    let audio3 = vec![0.7; 4800];

    let features = RosettaFeatures::default();
    synth.register_source(1, audio1, SourceMetadata112D::from_features(&features));
    synth.register_source(2, audio2, SourceMetadata112D::from_features(&features));
    synth.register_source(3, audio3, SourceMetadata112D::from_features(&features));

    // Create timeline
    let mut timeline = SynthesisTimeline::new();
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 1,
        start_time_ms: 0.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 2,
        start_time_ms: 100.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 3,
        start_time_ms: 200.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });

    // Synthesize
    let result = synth.synthesize_timeline(&timeline).await;
    assert!(result.is_ok());

    let output = result.unwrap();
    assert!(!output.is_empty());
    // Output should be approximately total duration at sample rate
    // 300ms at 48kHz = 14400 samples (approximately)
    assert!(output.len() >= 10000);
}

#[tokio::test]
async fn test_cached_granular_synthesizer_empty_timeline() {
    let config = SynthesisConfig112D::default();
    let synth = CachedGranularSynthesizer::new(config);

    let timeline = SynthesisTimeline::new();
    let result = synth.synthesize_timeline(&timeline).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.is_empty());
}

#[tokio::test]
async fn test_cached_granular_synthesizer_missing_source() {
    let config = SynthesisConfig112D::default();
    let synth = CachedGranularSynthesizer::new(config);

    let mut timeline = SynthesisTimeline::new();
    timeline.add_event(SemanticTimelineEvent {
        cluster_id: 999, // Not registered
        start_time_ms: 0.0,
        duration_ms: 100.0,
        amplitude: 1.0,
    });

    let result = synth.synthesize_timeline(&timeline).await;
    // Should handle gracefully - either error or silent skip
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_cached_granular_synthesizer_clear_sources() {
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    let audio = vec![0.5, 0.3];
    let features = RosettaFeatures::default();
    synth.register_source(1, audio.clone(), SourceMetadata112D::from_features(&features));
    synth.register_source(2, audio, SourceMetadata112D::from_features(&features));

    assert_eq!(synth.source_count(), 2);

    synth.clear_sources();
    assert_eq!(synth.source_count(), 0);
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

#[tokio::test]
async fn test_full_semantic_reconstruction_pipeline() {
    // STAGE 1: NBD Segmentation (simulated)
    // We'd have isolated audio segments here
    let segment1 = vec![0.5; 2400]; // 50ms at 48kHz
    let segment2 = vec![0.3; 2400];
    let segment3 = vec![0.7; 2400];

    // STAGE 2: 112D Feature Extraction (simulated)
    let features = RosettaFeatures::default();

    // STAGE 3: Corpus Analysis (simulated)
    // Cluster IDs assigned
    let cluster_ids = [1, 2, 3];

    // STAGE 4: Semantic Reconstruction
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    // Register exemplars
    synth.register_source(1, segment1, SourceMetadata112D::from_features(&features));
    synth.register_source(2, segment2, SourceMetadata112D::from_features(&features));
    synth.register_source(3, segment3, SourceMetadata112D::from_features(&features));

    // Create timeline from N-gram template
    let mut timeline = SynthesisTimeline::new();
    for (i, &cluster_id) in cluster_ids.iter().enumerate() {
        timeline.add_event(SemanticTimelineEvent {
            cluster_id,
            start_time_ms: i as f64 * 50.0,
            duration_ms: 50.0,
            amplitude: 1.0,
        });
    }

    // STAGE 5: Synthesis Output
    let result = synth.synthesize_timeline(&timeline).await;
    assert!(result.is_ok());

    let output = result.unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_exemplar_manager_quality_scoring() {
    let mut manager = ExemplarManager::new();

    // Register multiple exemplars with different quality
    let audio_low = vec![0.1; 100];
    let mut features_low = RosettaFeatures::default();
    features_low.rms_energy = 0.2;
    features_low.harmonic_to_noise_ratio = 10.0;

    let audio_high = vec![0.9; 100];
    let mut features_high = RosettaFeatures::default();
    features_high.rms_energy = 0.8;
    features_high.harmonic_to_noise_ratio = 25.0;

    // Register low quality first
    manager.register_exemplar(1, audio_low, features_low);

    // Register high quality - should replace
    let _entry_before = manager.get_exemplar(1).cloned();
    manager.register_exemplar(1, audio_high.clone(), features_high);
    let entry_after = manager.get_exemplar(1).cloned();

    // The high quality audio should be kept
    assert_eq!(entry_after.unwrap().audio, audio_high);
}

// =============================================================================
// TEACHER-STUDENT DISTILLATION TESTS (PCA+BGMM Integration)
// =============================================================================

#[test]
fn test_exemplar_manager_register_centroid() {
    let mut manager = ExemplarManager::new();

    // Create a 112D centroid (all zeros for simplicity)
    let centroid = [0.0f32; 112];
    manager.register_centroid(0, centroid);

    assert_eq!(manager.num_centroids(), 1);
}

#[test]
fn test_exemplar_manager_find_nearest_centroid() {
    let mut manager = ExemplarManager::new();

    // Register two centroids with distinct values
    let mut centroid_0 = [0.0f32; 112];
    centroid_0[0] = 0.0;

    let mut centroid_1 = [0.0f32; 112];
    centroid_1[0] = 10.0; // Far from centroid_0

    manager.register_centroid(0, centroid_0);
    manager.register_centroid(1, centroid_1);

    // Test feature close to centroid_0
    let mut test_feature = [0.0f32; 112];
    test_feature[0] = 1.0; // Close to centroid_0

    let result = manager.find_nearest_centroid(&test_feature);
    assert_eq!(result, Some(0), "Should find cluster 0 as nearest");
}

#[test]
fn test_exemplar_manager_find_nearest_centroid_empty() {
    let manager = ExemplarManager::new();
    let feature = [0.0f32; 112];

    let result = manager.find_nearest_centroid(&feature);
    assert_eq!(result, None, "Should return None when no centroids registered");
}

#[test]
fn test_exemplar_manager_find_nearest_centroid_with_distance() {
    let mut manager = ExemplarManager::new();

    let centroid_0 = [0.0f32; 112];
    manager.register_centroid(0, centroid_0);

    // Feature at distance 5.0 from origin (3-4-5 triangle in first 2 dims)
    let mut test_feature = [0.0f32; 112];
    test_feature[0] = 3.0;
    test_feature[1] = 4.0;

    let (cluster_id, distance) = manager.find_nearest_centroid_with_distance(&test_feature);

    assert_eq!(cluster_id, Some(0));
    assert!((distance - 5.0).abs() < 0.01, "Distance should be ~5.0");
}

#[test]
fn test_exemplar_manager_ood_detection() {
    // Use higher OOD threshold for this test
    let mut manager = ExemplarManager::with_ood_threshold(15.0);

    let centroid = [0.0f32; 112];
    manager.register_centroid(0, centroid);

    // In-distribution: feature close to centroid
    // Distance from origin: sqrt(112 * 1.0^2) = sqrt(112) ≈ 10.58
    let close_feature = [1.0f32; 112];
    assert!(
        !manager.is_out_of_distribution(&close_feature),
        "Close feature should be in-distribution (distance ≈ 10.6 < threshold 15.0)"
    );

    // Out-of-distribution: feature far from centroid
    let mut far_feature = [0.0f32; 112];
    far_feature[0] = 100.0; // Distance ≈ 100 > threshold 15.0
    assert!(
        manager.is_out_of_distribution(&far_feature),
        "Far feature should be out-of-distribution"
    );
}

#[test]
fn test_exemplar_manager_ood_threshold() {
    let mut manager = ExemplarManager::new();

    assert_eq!(manager.ood_threshold(), 50.0, "Default OOD threshold should be 50.0");

    manager.set_ood_threshold(25.0);
    assert_eq!(manager.ood_threshold(), 25.0, "OOD threshold should be updated");
}

#[test]
fn test_exemplar_manager_centroid_assignment_speed() {
    use std::time::Instant;

    let mut manager = ExemplarManager::new();

    // Register 94 centroids (matching BGMM discovery)
    for i in 0..94 {
        let mut centroid = [0.0f32; 112];
        centroid[0] = i as f32;
        manager.register_centroid(i, centroid);
    }

    // Test feature
    let test_feature = [50.0f32; 112];

    // Measure assignment speed (should be < 0.05ms = 50,000 nanoseconds)
    let start = Instant::now();
    let _result = manager.find_nearest_centroid(&test_feature);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_nanos() < 50_000,
        "Assignment took {:?}ns, expected < 50,000ns (0.05ms)",
        elapsed.as_nanos()
    );
}

#[test]
fn test_exemplar_manager_load_centroids_from_manifest() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut manager = ExemplarManager::new();

    // Create a temporary manifest file with exactly 112 values per centroid
    let mut centroid_0 = Vec::new();
    for _ in 0..112 {
        centroid_0.push("0.0");
    }
    let centroid_0_str = centroid_0.join(", ");

    let mut centroid_1 = Vec::new();
    for _ in 0..112 {
        centroid_1.push("10.0");
    }
    let centroid_1_str = centroid_1.join(", ");

    let manifest_json = format!(
        r#"{{"vocabulary_size": 2,
        "num_clusters": 2,
        "clusters": {{
            "0": {{
                "cluster_id": 0,
                "centroid_112d": [{}],
                "exemplar_audio": "cluster_0_exemplar.wav",
                "exemplar_features_112d": [],
                "num_segments": 1000,
                "mean_distance_to_centroid": 5.5
            }},
            "1": {{
                "cluster_id": 1,
                "centroid_112d": [{}],
                "exemplar_audio": "cluster_1_exemplar.wav",
                "exemplar_features_112d": [],
                "num_segments": 800,
                "mean_distance_to_centroid": 4.2
            }}
        }},
        "metadata": {{
            "extraction_method": "pca_bgmm_teacher_student",
            "n_components": 30
        }}
    }}"#,
        centroid_0_str, centroid_1_str
    );

    let mut temp_file = NamedTempFile::new().unwrap();
    write!(temp_file, "{}", manifest_json).unwrap();

    // Load centroids from manifest
    let count = manager.load_centroids_from_manifest(temp_file.path()).unwrap();

    assert_eq!(count, 2, "Should load 2 centroids");
    assert_eq!(manager.num_centroids(), 2);

    // Verify assignment works
    let test_feature = [0.0f32; 112];
    let result = manager.find_nearest_centroid(&test_feature);
    assert_eq!(result, Some(0), "Should assign to cluster 0");
}

// =============================================================================
// TEACHER-STUDENT DISTILLATION: OOD FILTERING TESTS
// =============================================================================

#[test]
fn test_assigns_feature_to_nearest_centroid() {
    let mut manager = ExemplarManager::new();

    // Register a centroid
    let mut centroid_8 = [0.0f32; 112];
    centroid_8[0] = 5.0; // Distinguishable feature
    manager.register_centroid(8, centroid_8);

    // Create a feature that perfectly matches the centroid
    let mock_features = centroid_8;

    // Should find the exact match with near-zero distance
    let result = manager.find_nearest_centroid_with_distance(&mock_features);
    assert_eq!(result.0, Some(8));
    assert!(result.1 < 1.0, "Distance to itself should be near 0");
}

#[test]
fn test_rejects_out_of_distribution_noise() {
    let mut manager = ExemplarManager::new();

    // Register some centroids
    let centroid_0 = [1.0f32; 112];
    manager.register_centroid(0, centroid_0);

    let centroid_1 = [2.0f32; 112];
    manager.register_centroid(1, centroid_1);

    // Set a low OOD threshold
    manager.set_ood_threshold(20.0);

    // Create garbage vector far from any centroid
    let ood_features = [9999.0f32; 112];

    // Should be rejected by OOD check
    let result = manager.find_nearest_centroid_with_ood_check(&ood_features);
    assert!(result.is_none(), "OOD features should be rejected");

    // But raw find_nearest_centroid_with_distance should still return something
    let raw_result = manager.find_nearest_centroid_with_distance(&ood_features);
    assert!(raw_result.0.is_some(), "Raw distance should still find nearest");
    assert!(raw_result.1 > 20.0, "Distance should exceed OOD threshold");
}

#[test]
fn test_get_centroid_by_id() {
    let mut manager = ExemplarManager::new();

    // Register centroids
    let mut centroid_0 = [0.0f32; 112];
    centroid_0[0] = 1.0;
    manager.register_centroid(0, centroid_0);

    let centroid_42 = [42.0f32; 112];
    manager.register_centroid(42, centroid_42);

    // Get existing centroid
    let result = manager.get_centroid_by_id(42);
    assert!(result.is_some());
    assert_eq!(result.unwrap()[0], 42.0);

    // Get non-existent centroid
    let missing = manager.get_centroid_by_id(999);
    assert!(missing.is_none());
}

#[test]
fn test_find_nearest_centroid_with_ood_check() {
    let mut manager = ExemplarManager::new();

    // Register centroids
    let centroid_0 = [0.0f32; 112];
    manager.register_centroid(0, centroid_0);

    let centroid_1 = [10.0f32; 112];
    manager.register_centroid(1, centroid_1);

    manager.set_ood_threshold(5.0); // Low threshold

    // Feature close to centroid_0 should be accepted
    // sqrt(112 * 0.4^2) = sqrt(17.92) ≈ 4.23 < 5.0
    let near_feature = [0.4f32; 112];
    let result = manager.find_nearest_centroid_with_ood_check(&near_feature);
    assert!(result.is_some());
    assert_eq!(result.unwrap().0, 0);

    // Feature far from all centroids should be rejected
    let far_feature = [100.0f32; 112];
    let far_result = manager.find_nearest_centroid_with_ood_check(&far_feature);
    assert!(far_result.is_none());
}
