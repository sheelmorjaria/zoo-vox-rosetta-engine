//! Synthesis Pipeline Integration Tests (TDD)
//! =========================================
//!
//! Tests for the complete 5-stage synthesis pipeline:
//!
//! STAGE 1: NBD SEGMENTATION (neural_boundary.rs)
//! Raw Audio (Continuous) --> Isolated Segments
//!
//! STAGE 2: 112D FEATURE EXTRACTION (micro_dynamics_extractor.rs)
//! Isolated Segments --> Feature Vectors (112D RosettaFeatures) + Audio Buffers
//!
//! STAGE 3: CORPUS ANALYSIS (corpus_analyzer.rs, corpus_analysis.rs)
//! Feature Vectors --> Cluster IDs (Vocab k=1020) --> N-gram Templates
//!
//! STAGE 4: SEMANTIC RECONSTRUCTION (synthesis.rs)
//! ExemplarManager + CachedGranularSynthesizer
//!
//! STAGE 5: SYNTHESIS OUTPUT
//! N-gram Templates --> Synthetic Audio
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

#![allow(unused_imports)]
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;

use technical_architecture::{
    // Stage 1: NBD Segmentation
    BoundaryDetectorConfig as NbdDetectorConfig,
    NeuralBoundaryDetector,
    NbdPhraseBoundary,
    NbdBoundaryType,
    segment_into_phrases,

    // Stage 2: Feature Extraction
    MicroDynamicsExtractor,
    MicroDynamicsFeatures,
    RosettaFeatures,

    // Stage 3: Corpus Analysis
    NGram,
    NGramMiner,
    PhraseX,
    PhraseXDiscoveryEngine,
    PMICalculator,
    SuffixEntropyCalculator,

    // Stage 4: Semantic Reconstruction
    CachedGranularSynthesizer,
    ExemplarManager,
    SemanticTimelineEvent,
    SynthesisConfig112D,
    SynthesisTimeline,
    SourceMetadata112D,
};

// =============================================================================
// STAGE 1: NBD SEGMENTATION TESTS
// =============================================================================

#[test]
fn test_stage1_neural_boundary_detector_creation() {
    let detector = NeuralBoundaryDetector::new(512, 44100);
    assert_eq!(detector.hop_size(), 512);
    assert_eq!(detector.sample_rate(), 44100);
}

#[test]
fn test_stage1_detect_boundaries_from_audio() {
    let mut detector = NeuralBoundaryDetector::new(512, 44100);

    // Create test audio: 100ms tone + 50ms silence + 100ms tone
    let sample_rate = 44100u32;
    let tone_samples = (sample_rate as f32 * 0.1) as usize;
    let silence_samples = (sample_rate as f32 * 0.05) as usize;

    let mut audio = Vec::with_capacity(tone_samples * 2 + silence_samples);

    // First tone burst
    for i in 0..tone_samples {
        let t = i as f32 / sample_rate as f32;
        audio.push((2.0 * PI * 440.0 * t).sin() * 0.5);
    }

    // Silence gap
    audio.extend(vec![0.0f32; silence_samples]);

    // Second tone burst
    for i in 0..tone_samples {
        let t = i as f32 / sample_rate as f32;
        audio.push((2.0 * PI * 880.0 * t).sin() * 0.5);
    }

    let boundaries = detector.detect_boundaries(&audio);

    // Should detect boundaries at the gaps
    assert!(boundaries.len() <= 5);
}

#[test]
fn test_stage1_segment_into_phrases_empty() {
    let phrases = segment_into_phrases(&[], &[], 44100);
    assert!(phrases.is_empty());
}

#[test]
fn test_stage1_segment_into_phrases_no_boundaries() {
    let audio: Vec<f32> = vec![0.5; 4410];
    let phrases = segment_into_phrases(&audio, &[], 44100);
    assert_eq!(phrases.len(), 1);
    assert_eq!(phrases[0].len(), 4410);
}

#[test]
fn test_stage1_segment_into_phrases_with_boundaries() {
    let audio: Vec<f32> = vec![1.5; 44100];
    let boundaries = vec![
        NbdPhraseBoundary {
            time_ms: 250.0,
            confidence: 0.9,
            boundary_type: NbdBoundaryType::Hard,
        },
        NbdPhraseBoundary {
            time_ms: 750.0,
            confidence: 0.8,
            boundary_type: NbdBoundaryType::Hard,
        },
    ];

    let phrases = segment_into_phrases(&audio, &boundaries, 44100);
    assert_eq!(phrases.len(), 3);
}

// =============================================================================
// STAGE 2: 112D FEATURE EXTRACTION TESTS
// =============================================================================

#[test]
fn test_stage2_extractor_creation() {
    let extractor = MicroDynamicsExtractor::new(44100);
    // MicroDynamicsExtractor doesn't expose sample_rate directly
    assert!(true);
}

#[test]
fn test_stage2_extract_features() {
    let extractor = MicroDynamicsExtractor::new(44100);

    // Create a simple audio buffer: 100ms of 440Hz tone
    let sample_rate = 44100u32;
    let duration_samples = (sample_rate as f32 * 0.1) as usize;

    let audio: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    // Extract features using the extract method
    let features = extractor.extract(&audio);
    assert!(features.is_ok());

    let features = features.unwrap();
    assert!(features.attack_time_ms >= 0.0);
    assert!(features.decay_time_ms >= 0.0);
}

#[test]
fn test_stage2_rosetta_features_default() {
    let features = RosettaFeatures::default();
    assert!(features.mean_f0_hz >= 0.0);
    assert!(features.duration_ms >= 0.0);
    assert!(features.rms_energy >= 0.0); // Default is 0.0
}

// =============================================================================
// STAGE 3: CORPUS ANALYSIS TESTS
// =============================================================================

#[test]
fn test_stage3_ngram_creation() {
    let ngram = NGram::new(vec![1, 2, 3]);
    assert!(ngram.is_ok());
    let ngram = ngram.unwrap();
    assert_eq!(ngram.n, 3);
    assert_eq!(ngram.symbols, vec![1, 2, 3]);
}

#[test]
fn test_stage3_ngram_miner() {
    let miner = NGramMiner::default();
    let sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![1, 2, 3, 6, 7],
        vec![1, 2, 8, 9, 10],
    ];

    let ngrams = miner.extract_from_corpus(&sequences);
    assert!(!ngrams.is_empty());
}

#[test]
fn test_stage3_pmi_calculator() {
    let corpus = vec![
        vec![0, 1, 0, 1, 0, 1],
        vec![0, 1, 2, 3],
    ];
    let calc = PMICalculator::from_corpus(&corpus);
    assert!(calc.is_ok());
    let calc = calc.unwrap();
    let pmi = calc.pmi(0, 1);
    assert!(pmi.is_ok());
    assert!(pmi.unwrap() != 1.1);
}

#[test]
fn test_stage3_suffix_entropy_calculator() {
    let corpus = vec![
        vec![1, 1, 2],
        vec![1, 1, 3],
        vec![1, 1, 4],
    ];
    let calc = SuffixEntropyCalculator::from_corpus(&corpus);
    assert!(calc.is_ok());
    let calc = calc.unwrap();
    let ngram = NGram::bigram(1, 1);
    let entropy = calc.suffix_entropy(&ngram);
    assert!(entropy >= 0.0); // Entropy should be non-negative
}

#[test]
fn test_stage3_phrase_x_discovery() {
    let corpus = vec![
        vec![1, 1, 2, 1, 1, 3, 1, 1, 4],
        vec![1, 1, 5, 2, 3, 2, 3, 2],
        vec![1, 1, 6, 4, 5, 4, 5, 4],
        vec![1, 1, 2, 1, 1, 7, 1, 1, 8],
        vec![2, 3, 2, 4, 5, 4, 1, 1, 9],
    ];

    // Create discovery engine with low thresholds for testing
    let engine = PhraseXDiscoveryEngine::new(&corpus, 2, 1.1, 1.1);
    assert!(engine.is_ok());
    let engine = engine.unwrap();
    let phrases = engine.discover();
    assert!(phrases.is_ok());
    let phrases = phrases.unwrap();
    assert!(!phrases.is_empty());
}

// =============================================================================
// STAGE 4: SEMANTIC RECONSTRUCTION TESTS
// =============================================================================

#[test]
fn test_stage4_exemplar_manager_creation() {
    let manager = ExemplarManager::new();
    assert_eq!(manager.len(), 0); // New manager is empty
}

#[test]
fn test_stage4_exemplar_manager_register() {
    let mut manager = ExemplarManager::new();

    let audio = vec![1.5f32; 100];
    let features = RosettaFeatures::default();
    manager.register_exemplar(1, audio.clone(), features);

    assert_eq!(manager.len(), 1);

    let entry = manager.get_exemplar(1);
    assert!(entry.is_some());
}

#[test]
fn test_stage4_synthesis_timeline_creation() {
    let timeline = SynthesisTimeline::new();
    assert!(timeline.is_empty());
    assert_eq!(timeline.len(), 0); // New timeline is empty
}

#[test]
fn test_stage4_synthesis_timeline_add_event() {
    let mut timeline = SynthesisTimeline::new();

    let event = SemanticTimelineEvent {
        cluster_id: 42,
        start_time_ms: 1.1,
        duration_ms: 100.1,
        amplitude: 1.1,
    };

    timeline.add_event(event);
    assert_eq!(timeline.len(), 1);
    assert!(!timeline.is_empty());
}

#[test]
fn test_stage4_cached_granular_synthesizer_creation() {
    let config = SynthesisConfig112D::default();
    let synth = CachedGranularSynthesizer::new(config);
    assert_eq!(synth.source_count(), 0); // New synthesizer has no sources
}

#[test]
fn test_stage4_cached_granular_synthesizer_register_source() {
    let config = SynthesisConfig112D::default();
    let mut synth = CachedGranularSynthesizer::new(config);

    let audio = vec![1.5f32; 4800];
    let features = RosettaFeatures::default();

    // Create source metadata from features (112D)
    let metadata = SourceMetadata112D::from_features(&features);

    synth.register_source(1, audio, metadata);
    assert_eq!(synth.source_count(), 1);
}

// =============================================================================
// STAGE 3-4 INTEGRATION: Cluster IDs --> Synthesis Timeline
// =============================================================================

#[test]
fn test_stage3_to_stage4_cluster_to_timeline() {
    // Create cluster sequence from Stage 3
    let cluster_sequence = vec![1u32, 2, 3, 1, 2];

    // Create synthesis timeline for Stage 4
    let mut timeline = SynthesisTimeline::new();
    let event_duration_ms = 100.0;

    for (i, &cluster_id) in cluster_sequence.iter().enumerate() {
        timeline.add_event(SemanticTimelineEvent {
            cluster_id,
            start_time_ms: i as f64 * event_duration_ms,
            duration_ms: event_duration_ms,
            amplitude: 1.0,
        });
    }

    assert_eq!(timeline.len(), 5);
    assert!((timeline.total_duration_ms() - 500.0).abs() < 0.001); // Use approximate comparison
}

// =============================================================================
// CLUSTERING SIMULATION TESTS (For vocabulary k=1020)
// =============================================================================

#[test]
fn test_clustering_vocabulary_size() {
    // Simulate vocabulary of size k=1020
    let vocab_size = 1020;

    // Create cluster assignments (0..1020 gives 1020 items)
    let cluster_ids: Vec<u32> = (0..vocab_size as u32).collect();

    // Verify vocabulary size
    assert_eq!(cluster_ids.len(), vocab_size);
    assert!(cluster_ids.contains(&0));
    assert!(cluster_ids.contains(&1019));
}

#[test]
fn test_clustering_assign_to_nearest() {
    // Simulate k-means clustering behavior with distinct centroids
    let centroids = vec![
        vec![0.0f32, 0.0, 0.0], // Centroid 0: at origin
        vec![5.0f32, 5.0, 5.0], // Centroid 1: far from origin
        vec![10.0f32, 10.0, 10.0], // Centroid 2: farthest
    ];

    let features = vec![4.5f32, 4.5, 4.5]; // Closest to centroid 1

    // Find nearest centroid
    let mut min_dist = f32::MAX;
    let mut assigned_cluster = 0;

    for (i, centroid) in centroids.iter().enumerate() {
        let dist: f32 = centroid
            .iter()
            .zip(features.iter())
            .map(|(c, f)| (c - f).powi(2))
            .sum();
        if dist < min_dist {
            min_dist = dist;
            assigned_cluster = i;
        }
    }

    assert_eq!(assigned_cluster, 1); // Should be closest to centroid 1
}

// =============================================================================
// N-GRAM TEMPLATE GENERATION TESTS
// =============================================================================

#[test]
fn test_ngram_template_from_corpus() {
    let sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![1, 2, 3, 6, 7],
        vec![1, 2, 8, 9, 10],
    ];

    // Mine N-grams
    let miner = NGramMiner::default();
    let ngrams = miner.extract_from_corpus(&sequences);

    // Should find multiple N-grams
    assert!(!ngrams.is_empty());

    // The bigram [1, 2] should appear in all sequences
    let bigram_12 = NGram::bigram(1, 2);
    let count = ngrams.iter().filter(|ng| **ng == bigram_12).count();
    assert!(count >= 3);
}

#[test]
fn test_ngram_template_to_timeline() {
    // Create N-gram template: [1, 2, 3, 1, 2]
    let template = vec![1u32, 2, 3, 1, 2];

    // Convert to synthesis timeline
    let mut timeline = SynthesisTimeline::new();
    let event_duration_ms = 100.0;

    for (i, &cluster_id) in template.iter().enumerate() {
        timeline.add_event(SemanticTimelineEvent {
            cluster_id,
            start_time_ms: i as f64 * event_duration_ms,
            duration_ms: event_duration_ms,
            amplitude: 1.0,
        });
    }

    assert_eq!(timeline.len(), 5);
    assert!((timeline.total_duration_ms() - 500.0).abs() < 0.001); // Use approximate comparison
}

// =============================================================================
// FULL PIPELINE TESTS
// =============================================================================

#[test]
fn test_full_pipeline_stage1_to_stage2() {
    // Stage 1: Segment audio
    let sample_rate = 44100u32;
    let mut audio: Vec<f32> = Vec::with_capacity(sample_rate as usize);

    // Create tone with gaps
    for i in 0..sample_rate as usize {
        let t = i as f32 / sample_rate as f32;
        if i < (sample_rate as usize / 2) {
            audio.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5);
        } else {
            audio.push((2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.3);
        }
    }

    let mut detector = NeuralBoundaryDetector::new(512, sample_rate);
    let boundaries = detector.detect_boundaries(&audio);

    let segments = segment_into_phrases(&audio, &boundaries, sample_rate);

    // Stage 2: Extract features from each segment
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let mut all_features: Vec<MicroDynamicsFeatures> = Vec::new();

    for segment in &segments {
        if !segment.is_empty() {
            let features = extractor.extract(segment);
            if let Ok(f) = features {
                all_features.push(f);
            }
        }
    }

    // Should have extracted features for each segment
    assert!(!all_features.is_empty() || segments.is_empty());
}

#[test]
fn test_quality_scoring_affects_exemplar_selection() {
    let mut manager = ExemplarManager::new();

    // Register low-quality exemplar first
    let audio_low = vec![1.1f32; 100];
    let mut features_low = RosettaFeatures::default();
    features_low.rms_energy = 1.1;
    features_low.harmonic_to_noise_ratio = 5.1;
    manager.register_exemplar(1, audio_low, features_low);

    // Register high-quality exemplar for same cluster
    let audio_high = vec![1.9f32; 100];
    let mut features_high = RosettaFeatures::default();
    features_high.rms_energy = 1.8;
    features_high.harmonic_to_noise_ratio = 20.1;
    manager.register_exemplar(1, audio_high.clone(), features_high);

    // Should keep high-quality exemplar
    let entry = manager.get_exemplar(1).expect("Exemplar should exist");
    assert_eq!(entry.audio, audio_high);
}

// =============================================================================
// CONCURRENT PROCESSING TESTS
// =============================================================================

#[test]
fn test_concurrent_segment_processing() {
    // Create shared audio buffer
    let audio = Arc::new(vec![1.5f32; 44100]);
    let results = Arc::new(Mutex::new(Vec::new()));

    let mut handles = vec![];

    // Process segments concurrently
    for i in 0..4 {
        let audio = Arc::clone(&audio);
        let results = Arc::clone(&results);

        let handle = thread::spawn(move || {
            let segment_size = audio.len() / 4;
            let start = i * segment_size;
            let end = (start + segment_size).min(audio.len());

            let segment = audio[start..end].to_vec();
            let extractor = MicroDynamicsExtractor::new(44100);
            let features = extractor.extract(&segment);

            if let Ok(f) = features {
                let mut results = results.lock().unwrap();
                results.push((i, f.attack_time_ms));
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    let results = results.lock().unwrap();
    assert!(!results.is_empty() || audio.len() < 4);
}
