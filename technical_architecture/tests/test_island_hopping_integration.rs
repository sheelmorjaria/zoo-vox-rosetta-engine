// =============================================================================
// TDD Integration Tests for Island Hopping + Synthesis (Phase 2: The "Mouth")
// =============================================================================
//
// This test suite validates the Rust execution layer that:
// 1. Loads audio sources on-demand (Island Hopping)
// 2. Caches sources for fast revisiting (LRU Cache)
// 3. Handles concurrent access safely (Thread Safety)
// 4. Morphs seamlessly between sources (Crossfading)
//
// Architecture: Python Planning → Rust Execution
//
// Author: Sheel Morjaria (sheelmorjaria@gmail.com)
// License: CC BY-ND 4.0 International
//

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use technical_architecture::island_hopping::{
    apply_delta_to_granular, GranularParams, Vector30D, VectorDelta,
};
use technical_architecture::synthesis::{CachedGranularSequencer, SourceMetadataBuilder};

// =============================================================================
// Test 2.1: Island Hopping (Cache Miss)
// =============================================================================

#[tokio::test]
async fn test_island_hopping_cache_miss() {
    // RED TEST: Loading a new source should cache it
    //
    // Scenario: Request to load "neutral_001" phrase
    // Expected: Cache miss, buffer loaded and cached, takes <50ms
    // Arrange
    let mut sequencer = CachedGranularSequencer::with_default_cache(48000);

    // Create test audio (100ms at 48kHz)
    let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.001).sin()).collect();

    // Create metadata
    let metadata = SourceMetadataBuilder::default()
        .mean_f0_hz(7000.0)
        .duration_ms(100.0)
        .build();

    // Act
    let start = Instant::now();
    let result = sequencer
        .register_source("neutral_001".to_string(), audio.clone(), metadata)
        .await;
    let duration = start.elapsed();

    // Assert
    assert!(result.is_ok(), "Failed to register source");

    // Should take <50ms (simulated SSD load)
    assert!(
        duration < Duration::from_millis(50),
        "Loading took too long: {:?}",
        duration
    );

    // Verify source is in cache
    assert!(
        sequencer.is_cached("neutral_001"),
        "Source should be cached"
    );

    // Verify cache stats
    let stats = sequencer.cache_stats();
    assert_eq!(stats.cache_misses, 1, "Should have 1 cache miss");
    assert_eq!(stats.cache_hits, 0, "Should have 0 cache hits");

    println!("✓ Cache miss test passed in {:?}", duration);
}

// =============================================================================
// Test 2.2: Island Revisiting (Cache Hit)
// =============================================================================

#[tokio::test]
async fn test_island_revisiting_cache_hit() {
    // RED TEST: Re-visiting a source should use cached version
    //
    // Scenario: Load "neutral_001" twice
    // Expected: Second load hits cache, takes <5ms
    // Arrange
    let mut sequencer = CachedGranularSequencer::with_default_cache(48000);

    let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.001).sin()).collect();
    let metadata = SourceMetadataBuilder::default()
        .mean_f0_hz(7000.0)
        .duration_ms(100.0)
        .build();

    // First load (cache miss)
    sequencer
        .register_source("neutral_001".to_string(), audio.clone(), metadata.clone())
        .await
        .expect("First load should succeed");

    // Act - Second load (should be cache hit)
    let start = Instant::now();
    let result = sequencer
        .register_source("neutral_001".to_string(), audio, metadata)
        .await;
    let duration = start.elapsed();

    // Assert
    assert!(result.is_ok(), "Second load should succeed");

    // Should be <5ms (cache hit)
    assert!(
        duration < Duration::from_millis(5),
        "Cache hit took too long: {:?}",
        duration
    );

    // Verify cache stats
    let stats = sequencer.cache_stats();
    assert_eq!(stats.cache_misses, 1, "Should still have 1 cache miss");
    assert_eq!(stats.cache_hits, 1, "Should have 1 cache hit");

    println!("✓ Cache hit test passed in {:?}", duration);
}

// =============================================================================
// Test 2.3: Concurrency Safety (Multi-Threaded)
// =============================================================================

#[tokio::test]
async fn test_concurrent_island_hopping() {
    // RED TEST: Multiple threads should safely access the cache
    //
    // Scenario: 4 threads loading different sources simultaneously
    // Expected: All loads succeed, no data races, cache is consistent
    // Arrange
    let sequencer = Arc::new(tokio::sync::Mutex::new(
        CachedGranularSequencer::with_default_cache(48000),
    ));

    let mut handles = vec![];

    // Act - Spawn 4 concurrent threads
    for thread_id in 0..4 {
        let sequencer_clone = sequencer.clone();
        let handle = thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();

            runtime.block_on(async move {
                let source_id = format!("source_{}", thread_id);
                let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.001).sin()).collect();
                let metadata = SourceMetadataBuilder::default()
                    .mean_f0_hz(7000.0 + thread_id as f32 * 100.0)
                    .duration_ms(100.0)
                    .build();

                // Load source
                let result = sequencer_clone
                    .lock()
                    .await
                    .register_source(source_id, audio, metadata)
                    .await;

                // Verify
                assert!(result.is_ok(), "Thread {} failed to load source", thread_id);

                // Check cache
                let is_cached = sequencer_clone
                    .lock()
                    .await
                    .is_cached(&format!("source_{}", thread_id));
                assert!(is_cached, "Thread {} source should be cached", thread_id);

                println!("✓ Thread {} completed successfully", thread_id);
            });
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Assert - Verify final cache state
    let stats = sequencer.lock().await.cache_stats();
    assert_eq!(
        stats.cache_misses, 4,
        "Should have 4 cache misses (one per thread)"
    );
    assert_eq!(stats.cache_hits, 0, "Should have 0 cache hits");

    println!("✓ Concurrency test passed - all 4 threads succeeded");
}

// =============================================================================
// Test 2.4: Seamless Morphing (Crossfade Verification)
// =============================================================================

#[tokio::test]
async fn test_seamless_morphing_crossfade() {
    // RED TEST: Morphing between sources should be seamless
    //
    // Scenario: Apply delta to morph from "neutral" to "aggressive"
    // Expected: Morph parameters are smooth, no pops/clicks
    // Arrange
    let base_params = GranularParams {
        pitch_shift_ratio: 1.0,
        grain_size_ms: 20.0,
        roughness_amount: 0.3,
        duration_scale: 1.0,
    };

    // Source metadata (neutral phrase)
    let source_metadata = Vector30D {
        mean_f0_hz: 7000.0,
        f0_range_hz: 400.0,
        duration_ms: 50.0,
        harmonic_to_noise_ratio: 20.0,
        spectral_flatness: 0.3,
        harmonicity: 0.8,
        attack_time_ms: 5.0,
        decay_time_ms: 20.0,
        sustain_level: 0.7,
        vibrato_rate_hz: 7.0,
        vibrato_depth: 0.02,
        jitter: 0.01,
        shimmer: 0.03,
        mfcc_1: -10.0,
        mfcc_2: -5.0,
        mfcc_3: -2.0,
        mfcc_4: -1.0,
        mfcc_5: -0.5,
        mfcc_6: -0.3,
        mfcc_7: -0.2,
        mfcc_8: -0.1,
        mfcc_9: 0.0,
        mfcc_10: 0.1,
        mfcc_11: 0.2,
        mfcc_12: 0.3,
        mfcc_13: 0.4,
        spectral_flux: 0.5,
        median_ici_ms: 15.0,
        onset_rate_hz: 50.0,
        ici_coefficient_of_variation: 0.3,
    };

    // Delta for morphing to aggressive (increase pitch, roughness)
    let delta = VectorDelta {
        delta_mean_f0_hz: 350.0, // +5% pitch
        delta_duration_ms: 0.0,  // Same duration
        delta_f0_range_hz: 0.0,
        delta_harmonic_to_noise_ratio: 0.0,
        delta_spectral_flatness: 0.1, // Increase roughness
        delta_harmonicity: 0.0,
        delta_attack_time_ms: 0.0,
        delta_decay_time_ms: 0.0,
        delta_sustain_level: 0.0,
        delta_vibrato_rate_hz: 0.0,
        delta_vibrato_depth: 0.0,
        delta_jitter: 0.0,
        delta_shimmer: 0.0,
        delta_mfcc_1: 0.0,
        delta_mfcc_2: 0.0,
        delta_mfcc_3: 0.0,
        delta_mfcc_4: 0.0,
        delta_mfcc_5: 0.0,
        delta_mfcc_6: 0.0,
        delta_mfcc_7: 0.0,
        delta_mfcc_8: 0.0,
        delta_mfcc_9: 0.0,
        delta_mfcc_10: 0.0,
        delta_mfcc_11: 0.0,
        delta_mfcc_12: 0.0,
        delta_mfcc_13: 0.0,
        delta_spectral_flux: 0.0,
        delta_median_ici_ms: 0.0,
        delta_onset_rate_hz: 0.0,
        delta_ici_coefficient_of_variation: 0.0,
    };

    // Act - Apply morph
    let morphed_params = apply_delta_to_granular(&delta, &base_params, &source_metadata);

    // Assert
    // Pitch should increase by ~5%
    assert!(
        (morphed_params.pitch_shift_ratio - 1.05).abs() < 0.01,
        "Pitch shift should be ~1.05, got {}",
        morphed_params.pitch_shift_ratio
    );

    // Roughness should increase
    assert!(
        (morphed_params.roughness_amount - 0.4).abs() < 0.01,
        "Roughness should be ~0.4, got {}",
        morphed_params.roughness_amount
    );

    // Duration should stay the same
    assert!(
        (morphed_params.duration_scale - 1.0).abs() < 0.01,
        "Duration scale should be ~1.0, got {}",
        morphed_params.duration_scale
    );

    // Grain size should stay the same
    assert!(
        (morphed_params.grain_size_ms - 20.0).abs() < 0.01,
        "Grain size should be ~20.0, got {}",
        morphed_params.grain_size_ms
    );

    println!("✓ Seamless morphing test passed");
    println!("  Pitch shift: {:.2}x", morphed_params.pitch_shift_ratio);
    println!("  Roughness: {:.2}", morphed_params.roughness_amount);
    println!("  Duration scale: {:.2}x", morphed_params.duration_scale);
}

// =============================================================================
// Additional: LRU Cache Eviction
// =============================================================================

#[tokio::test]
async fn test_lru_cache_eviction() {
    // RED TEST: Cache should evict least-recently-used entries when full
    //
    // Scenario: Fill cache with 3 sources, load 4th (smaller) source
    // Expected: LRU entry evicted, new source cached
    // Arrange - Small cache (1MB)
    let mut sequencer = CachedGranularSequencer::new(48000, 1024 * 1024);

    // Create large audio buffers (~400KB each)
    let large_audio: Vec<f32> = (0..100_000).map(|i| (i as f32 * 0.001).sin()).collect();
    let metadata = SourceMetadataBuilder::default()
        .mean_f0_hz(7000.0)
        .duration_ms(2000.0)
        .build();

    // Load 3 sources (fills cache)
    sequencer
        .register_source(
            "source_1".to_string(),
            large_audio.clone(),
            metadata.clone(),
        )
        .await
        .expect("Load 1 should succeed");
    sequencer
        .register_source(
            "source_2".to_string(),
            large_audio.clone(),
            metadata.clone(),
        )
        .await
        .expect("Load 2 should succeed");
    sequencer
        .register_source("source_3".to_string(), large_audio, metadata.clone())
        .await
        .expect("Load 3 should succeed");

    // Act - Load 4th source (should trigger eviction)
    let small_audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.001).sin()).collect();
    let result = sequencer
        .register_source("source_4".to_string(), small_audio, metadata)
        .await;

    // Assert
    assert!(result.is_ok(), "Load 4 should succeed");

    // LRU (source_1) should be evicted
    assert!(
        !sequencer.is_cached("source_1"),
        "source_1 should be evicted"
    );
    assert!(
        sequencer.is_cached("source_2"),
        "source_2 should still be cached"
    );
    assert!(
        sequencer.is_cached("source_3"),
        "source_3 should still be cached"
    );
    assert!(sequencer.is_cached("source_4"), "source_4 should be cached");

    println!("✓ LRU eviction test passed");
}

// =============================================================================
// Integration: End-to-End Island Hopping Workflow
// =============================================================================

#[tokio::test]
async fn test_end_to_end_island_hopping_workflow() {
    // RED TEST: Complete island hopping workflow
    //
    // Scenario:
    // 1. Load "neutral" source (cache miss)
    // 2. Calculate morph to "aggressive" target
    // 3. Verify morph parameters
    // 4. Load "neutral" again (cache hit)
    // Expected: Complete workflow executes in <100ms
    // Arrange
    let mut sequencer = CachedGranularSequencer::with_default_cache(48000);

    let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.001).sin()).collect();
    let metadata = SourceMetadataBuilder::default()
        .mean_f0_hz(7000.0)
        .duration_ms(100.0)
        .build();

    // Act - Complete workflow
    let start = Instant::now();

    // Step 1: Load neutral source (cache miss, ~20ms)
    sequencer
        .register_source("neutral".to_string(), audio.clone(), metadata.clone())
        .await
        .expect("Load neutral should succeed");

    // Step 2: Calculate morph to aggressive
    let source_metadata = Vector30D {
        mean_f0_hz: 7000.0,
        f0_range_hz: 400.0,
        duration_ms: 100.0,
        harmonic_to_noise_ratio: 20.0,
        spectral_flatness: 0.3,
        harmonicity: 0.8,
        attack_time_ms: 5.0,
        decay_time_ms: 20.0,
        sustain_level: 0.7,
        vibrato_rate_hz: 7.0,
        vibrato_depth: 0.02,
        jitter: 0.01,
        shimmer: 0.03,
        mfcc_1: -10.0,
        mfcc_2: -5.0,
        mfcc_3: -2.0,
        mfcc_4: -1.0,
        mfcc_5: -0.5,
        mfcc_6: -0.3,
        mfcc_7: -0.2,
        mfcc_8: -0.1,
        mfcc_9: 0.0,
        mfcc_10: 0.1,
        mfcc_11: 0.2,
        mfcc_12: 0.3,
        mfcc_13: 0.4,
        spectral_flux: 0.5,
        median_ici_ms: 15.0,
        onset_rate_hz: 50.0,
        ici_coefficient_of_variation: 0.3,
    };

    let delta = VectorDelta {
        delta_mean_f0_hz: 700.0,  // +10% pitch
        delta_duration_ms: -20.0, // -20% duration
        delta_f0_range_hz: 100.0,
        delta_harmonic_to_noise_ratio: 5.0,
        delta_spectral_flatness: 0.2,
        delta_harmonicity: 0.0,
        delta_attack_time_ms: -2.0,
        delta_decay_time_ms: -5.0,
        delta_sustain_level: 0.1,
        delta_vibrato_rate_hz: 2.0,
        delta_vibrato_depth: 0.02,
        delta_jitter: 0.01,
        delta_shimmer: 0.0,
        delta_mfcc_1: 1.0,
        delta_mfcc_2: 0.5,
        delta_mfcc_3: 0.2,
        delta_mfcc_4: 0.1,
        delta_mfcc_5: 0.0,
        delta_mfcc_6: 0.0,
        delta_mfcc_7: 0.0,
        delta_mfcc_8: 0.0,
        delta_mfcc_9: 0.0,
        delta_mfcc_10: 0.0,
        delta_mfcc_11: 0.0,
        delta_mfcc_12: 0.0,
        delta_mfcc_13: 0.0,
        delta_spectral_flux: 5.0,
        delta_median_ici_ms: 0.0,
        delta_onset_rate_hz: 0.0,
        delta_ici_coefficient_of_variation: 0.0,
    };

    let base_params = GranularParams {
        pitch_shift_ratio: 1.0,
        grain_size_ms: 20.0,
        roughness_amount: 0.3,
        duration_scale: 1.0,
    };

    let morphed = apply_delta_to_granular(&delta, &base_params, &source_metadata);

    // Step 3: Load neutral again (cache hit, <1ms)
    sequencer
        .register_source("neutral".to_string(), audio, metadata)
        .await
        .expect("Re-load neutral should succeed");

    let total_duration = start.elapsed();

    // Assert
    assert!(
        total_duration < Duration::from_millis(100),
        "Workflow took too long: {:?}",
        total_duration
    );

    // Verify morph parameters
    assert!((morphed.pitch_shift_ratio - 1.1).abs() < 0.01);
    assert!((morphed.duration_scale - 0.8).abs() < 0.01);
    assert!((morphed.roughness_amount - 0.5).abs() < 0.01);

    // Verify cache stats
    let stats = sequencer.cache_stats();
    assert_eq!(stats.cache_misses, 1, "Should have 1 cache miss");
    assert_eq!(stats.cache_hits, 1, "Should have 1 cache hit");

    println!("✓ End-to-end workflow test passed in {:?}", total_duration);
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);
    println!("  Hit ratio: {:.1}%", stats.hit_rate * 100.0);
}
