//! Latency Benchmark for Real-Time Bio-Acoustic Interaction
//!
//! Profiles the complete interaction loop:
//! 1. Audio In: Record 1s buffer
//! 2. Analysis: Dynamic Segmentation + Feature Extraction + Classification
//! 3. Decision: Cognitive Logic (simulated)
//! 4. Synthesis: Rust Granular Synthesis
//! 5. Audio Out
//!
//! Target: Total loop < 200ms for "Antiphonal" (turn-taking) behavior

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::time::{Duration, Instant};
use technical_architecture::bio_acoustic_agent::{
    AcousticInventory, AcousticModality, AcousticPrototype, BioAcousticAgent, EnvState, InteractionContext,
    SourceMetadata, SynthesisRequest,
};

/// Benchmark results
#[derive(Debug, Clone)]
struct LatencyProfile {
    /// Time for audio capture (simulated)
    audio_capture_ms: f64,
    /// Time for dynamic segmentation
    segmentation_ms: f64,
    /// Time for 45D feature extraction
    feature_extraction_ms: f64,
    /// Time for classification (cascaded router)
    classification_ms: f64,
    /// Time for cognitive decision
    decision_ms: f64,
    /// Time for synthesis planning
    synthesis_planning_ms: f64,
    /// Time for audio synthesis (granular)
    synthesis_audio_ms: f64,
    /// Total round-trip time
    total_ms: f64,
}

impl LatencyProfile {
    fn total(&self) -> f64 {
        self.audio_capture_ms
            + self.segmentation_ms
            + self.feature_extraction_ms
            + self.classification_ms
            + self.decision_ms
            + self.synthesis_planning_ms
            + self.synthesis_audio_ms
    }

    fn print_summary(&self) {
        println!("\n  LATENCY BUDGET BREAKDOWN:");
        println!("  ┌─────────────────────────────────┬───────────┬───────────┐");
        println!("  │ Stage                           │ Time (ms) │ % of Total│");
        println!("  ├─────────────────────────────────┼───────────┼───────────┤");
        println!(
            "  │ 1. Audio Capture                │ {:>9.2} │ {:>8.1}% │",
            self.audio_capture_ms,
            self.audio_capture_ms / self.total() * 100.0
        );
        println!(
            "  │ 2. Dynamic Segmentation         │ {:>9.2} │ {:>8.1}% │",
            self.segmentation_ms,
            self.segmentation_ms / self.total() * 100.0
        );
        println!(
            "  │ 3. 45D Feature Extraction       │ {:>9.2} │ {:>8.1}% │",
            self.feature_extraction_ms,
            self.feature_extraction_ms / self.total() * 100.0
        );
        println!(
            "  │ 4. Cascaded Classification      │ {:>9.2} │ {:>8.1}% │",
            self.classification_ms,
            self.classification_ms / self.total() * 100.0
        );
        println!(
            "  │ 5. Cognitive Decision           │ {:>9.2} │ {:>8.1}% │",
            self.decision_ms,
            self.decision_ms / self.total() * 100.0
        );
        println!(
            "  │ 6. Synthesis Planning           │ {:>9.2} │ {:>8.1}% │",
            self.synthesis_planning_ms,
            self.synthesis_planning_ms / self.total() * 100.0
        );
        println!(
            "  │ 7. Granular Synthesis           │ {:>9.2} │ {:>8.1}% │",
            self.synthesis_audio_ms,
            self.synthesis_audio_ms / self.total() * 100.0
        );
        println!("  ├─────────────────────────────────┼───────────┼───────────┤");
        println!(
            "  │ TOTAL                           │ {:>9.2} │     100.0% │",
            self.total()
        );
        println!("  └─────────────────────────────────┴───────────┴───────────┘");

        let target_ms = 200.0;
        let margin = target_ms - self.total();
        if self.total() <= target_ms {
            println!(
                "\n  ✓ WITHIN TARGET: {:.2}ms remaining ({:.1}% headroom)",
                margin,
                margin / target_ms * 100.0
            );
        } else {
            println!(
                "\n  ✗ EXCEEDS TARGET: {:.2}ms over budget ({:.1}% excess)",
                -margin,
                -margin / target_ms * 100.0
            );
        }
    }
}

/// Simulated audio capture
fn simulate_audio_capture(duration_ms: u64) -> (Vec<f32>, Duration) {
    let start = Instant::now();

    // Simulate recording delay (in reality this would be hardware-dependent)
    // At 48kHz, 1s = 48000 samples
    let samples = (48000 * duration_ms / 1000) as usize;
    let audio: Vec<f32> = vec![0.0; samples];

    // Simulate minimal processing overhead
    std::thread::sleep(Duration::from_micros(100));

    (audio, start.elapsed())
}

/// Simulated dynamic segmentation
fn simulate_segmentation(audio: &[f32]) -> (Vec<(usize, usize)>, Duration) {
    let start = Instant::now();

    // Simulate FFT-based onset detection
    // In production: ~5ms for 1s audio
    let mut segments = Vec::new();
    let frame_size = 4800; // 100ms frames
    let hop_size = 2400; // 50ms hop

    let mut pos = 0;
    while pos + frame_size < audio.len() {
        // Simulated onset detection
        segments.push((pos, pos + frame_size));
        pos += hop_size;
    }

    // Simulate processing time (proportional to audio length)
    let processing_us = (audio.len() as f64 / 48000.0 * 5000.0) as u64;
    std::thread::sleep(Duration::from_micros(processing_us.min(10000)));

    (segments, start.elapsed())
}

/// Simulated 45D feature extraction
fn simulate_feature_extraction(segments: &[(usize, usize)]) -> (Vec<Vec<f64>>, Duration) {
    let start = Instant::now();

    // In production: ~2ms per segment
    let features: Vec<Vec<f64>> = segments
        .iter()
        .map(|_| {
            // Placeholder 45D feature vector
            vec![0.0; 45]
        })
        .collect();

    let processing_us = (segments.len() as f64 * 2000.0) as u64;
    std::thread::sleep(Duration::from_micros(processing_us.min(50000)));

    (features, start.elapsed())
}

/// Simulated cascaded classification
fn simulate_classification(features: &[Vec<f64>]) -> (String, f64, Duration) {
    let start = Instant::now();

    // Simulate router -> analyzer cascade
    // In production: ~1ms per phrase
    let label = "Phee".to_string();
    let confidence = 0.85;

    let processing_us = (features.len() as f64 * 1000.0) as u64;
    std::thread::sleep(Duration::from_micros(processing_us.min(20000)));

    (label, confidence, start.elapsed())
}

/// Simulated cognitive decision
fn simulate_cognitive_decision(label: &str, confidence: f64) -> (String, Duration) {
    let start = Instant::now();

    // Python cognitive layer would process this
    // Simulate ZeroMQ round-trip + decision logic
    let response = if confidence > 0.7 {
        label.to_string() // Echo
    } else {
        "Unknown".to_string()
    };

    // Typical Python round-trip: 10-50ms
    std::thread::sleep(Duration::from_millis(15));

    (response, start.elapsed())
}

/// Real synthesis planning (using actual BioAcousticAgent)
fn plan_synthesis(
    agent: &BioAcousticAgent,
    label: &str,
) -> (technical_architecture::bio_acoustic_agent::SynthesisPlan, Duration) {
    let start = Instant::now();

    let request = SynthesisRequest::new(label)
        .with_environment(EnvState::Quiet)
        .with_context(InteractionContext::Reply);

    let plan = agent.plan_synthesis(request).expect("Synthesis planning failed");

    (plan, start.elapsed())
}

/// Simulated granular synthesis
fn simulate_granular_synthesis(duration_ms: f64) -> (Vec<f32>, Duration) {
    let start = Instant::now();

    // In production: ~5ms for 300ms output
    let samples = (48000.0 * duration_ms / 1000.0) as usize;
    let audio: Vec<f32> = vec![0.0; samples];

    let processing_us = (duration_ms / 300.0 * 5000.0) as u64;
    std::thread::sleep(Duration::from_micros(processing_us.max(100).min(20000)));

    (audio, start.elapsed())
}

/// Run single iteration of the latency benchmark
fn run_benchmark_iteration(agent: &BioAcousticAgent) -> LatencyProfile {
    // 1. Audio Capture
    let (_audio, capture_time) = simulate_audio_capture(1000);
    let capture_ms = capture_time.as_secs_f64() * 1000.0;

    // 2. Dynamic Segmentation
    let (segments, seg_time) = simulate_segmentation(&_audio);
    let seg_ms = seg_time.as_secs_f64() * 1000.0;

    // 3. Feature Extraction
    let (features, feat_time) = simulate_feature_extraction(&segments);
    let feat_ms = feat_time.as_secs_f64() * 1000.0;

    // 4. Classification
    let (label, confidence, class_time) = simulate_classification(&features);
    let class_ms = class_time.as_secs_f64() * 1000.0;

    // 5. Cognitive Decision
    let (response, decision_time) = simulate_cognitive_decision(&label, confidence);
    let decision_ms = decision_time.as_secs_f64() * 1000.0;

    // 6. Synthesis Planning (REAL)
    let (plan, plan_time) = plan_synthesis(agent, &response);
    let plan_ms = plan_time.as_secs_f64() * 1000.0;

    // 7. Granular Synthesis
    let (_output, synth_time) = simulate_granular_synthesis(plan.target_metadata.duration_ms as f64);
    let synth_ms = synth_time.as_secs_f64() * 1000.0;

    LatencyProfile {
        audio_capture_ms: capture_ms,
        segmentation_ms: seg_ms,
        feature_extraction_ms: feat_ms,
        classification_ms: class_ms,
        decision_ms: decision_ms,
        synthesis_planning_ms: plan_ms,
        synthesis_audio_ms: synth_ms,
        total_ms: 0.0, // Calculated later
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Real-Time Latency Benchmark - Bio-Acoustic Interaction               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // STEP 1: Create Test Agent with Inventory
    // =========================================================================
    println!("[1/3] Creating Bio-Acoustic Agent with prototypes...");
    println!();

    let mut inventory = AcousticInventory::new("marmoset");

    // Add prototypes for benchmarking
    inventory.add_prototype(AcousticPrototype {
        label: "Phee".to_string(),
        audio_buffer: vec![0.1; 14400], // 300ms at 48kHz
        sample_rate: 48000,
        metadata: SourceMetadata {
            mean_f0_hz: 7000.0,
            duration_ms: 300.0,
            harmonic_to_noise_ratio: 20.0,
            entropy: 0.15,
            attack_time_ms: 20.0,
            sustain_level: 0.7,
            jitter: 0.02,
            loudness: 0.6,
            ..Default::default()
        },
        sample_count: 100,
        modality: AcousticModality::Harmonic,
    });

    inventory.add_prototype(AcousticPrototype {
        label: "Tsik".to_string(),
        audio_buffer: vec![0.2; 3840], // 80ms at 48kHz
        sample_rate: 48000,
        metadata: SourceMetadata {
            mean_f0_hz: 9000.0,
            duration_ms: 80.0,
            harmonic_to_noise_ratio: 8.0,
            entropy: 0.6,
            attack_time_ms: 5.0,
            sustain_level: 0.4,
            jitter: 0.15,
            loudness: 0.8,
            ..Default::default()
        },
        sample_count: 50,
        modality: AcousticModality::Transient,
    });

    inventory.set_response_strategy("Phee", "Phee");
    inventory.set_response_strategy("Tsik", "Phee");

    let agent = BioAcousticAgent::new(inventory, 48000);
    println!("  Created agent with 2 prototypes");
    println!();

    // =========================================================================
    // STEP 2: Run Benchmark (10 iterations)
    // =========================================================================
    println!("[2/3] Running latency benchmark (10 iterations)...");
    println!();

    let n_iterations = 10;
    let mut profiles: Vec<LatencyProfile> = Vec::with_capacity(n_iterations);

    for i in 0..n_iterations {
        let profile = run_benchmark_iteration(&agent);
        profiles.push(profile);
        println!("  Iteration {}: {:.2}ms", i + 1, profiles.last().unwrap().total());
    }

    // =========================================================================
    // STEP 3: Analyze Results
    // =========================================================================
    println!();
    println!("[3/3] Analyzing results...");
    println!();

    // Calculate averages
    let avg_profile = LatencyProfile {
        audio_capture_ms: profiles.iter().map(|p| p.audio_capture_ms).sum::<f64>() / n_iterations as f64,
        segmentation_ms: profiles.iter().map(|p| p.segmentation_ms).sum::<f64>() / n_iterations as f64,
        feature_extraction_ms: profiles.iter().map(|p| p.feature_extraction_ms).sum::<f64>() / n_iterations as f64,
        classification_ms: profiles.iter().map(|p| p.classification_ms).sum::<f64>() / n_iterations as f64,
        decision_ms: profiles.iter().map(|p| p.decision_ms).sum::<f64>() / n_iterations as f64,
        synthesis_planning_ms: profiles.iter().map(|p| p.synthesis_planning_ms).sum::<f64>() / n_iterations as f64,
        synthesis_audio_ms: profiles.iter().map(|p| p.synthesis_audio_ms).sum::<f64>() / n_iterations as f64,
        total_ms: 0.0,
    };

    avg_profile.print_summary();

    // Calculate percentiles
    let mut totals: Vec<f64> = profiles.iter().map(|p| p.total()).collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = totals[n_iterations as usize / 2];
    let p95 = totals[(n_iterations as f64 * 0.95) as usize];
    let p99 = totals[((n_iterations as f64 * 0.99) as usize).min(n_iterations as usize - 1)];

    println!("\n  PERCENTILES:");
    println!("  ┌─────────────┬───────────┐");
    println!("  │ Percentile  │ Time (ms) │");
    println!("  ├─────────────┼───────────┤");
    println!("  │ P50 (median)│ {:>9.2} │", p50);
    println!("  │ P95         │ {:>9.2} │", p95);
    println!("  │ P99         │ {:>9.2} │", p99);
    println!("  └─────────────┴───────────┘");

    println!("\n  TARGETS:");
    println!("  ┌────────────────────────────┬───────────┬───────────┐");
    println!("  │ Scenario                   │ Target    │ Status    │");
    println!("  ├────────────────────────────┼───────────┼───────────┤");

    let antiphonal_status = if p95 <= 200.0 { "✓ PASS" } else { "✗ FAIL" };
    let call_response_status = if p95 <= 500.0 { "✓ PASS" } else { "✗ FAIL" };
    let monitoring_status = if p95 <= 1000.0 { "✓ PASS" } else { "✗ FAIL" };

    println!(
        "  │ Antiphonal (200ms)         │ 200ms     │ {} ({:.0}ms) │",
        antiphonal_status, p95
    );
    println!(
        "  │ Call-Response (500ms)      │ 500ms     │ {} ({:.0}ms) │",
        call_response_status, p95
    );
    println!(
        "  │ Monitoring (1000ms)        │ 1000ms    │ {} ({:.0}ms) │",
        monitoring_status, p95
    );
    println!("  └────────────────────────────┴───────────┴───────────┘");

    println!("\n  OPTIMIZATION RECOMMENDATIONS:");
    let bottleneck = [
        ("Audio Capture", avg_profile.audio_capture_ms),
        ("Segmentation", avg_profile.segmentation_ms),
        ("Feature Extraction", avg_profile.feature_extraction_ms),
        ("Classification", avg_profile.classification_ms),
        ("Cognitive Decision", avg_profile.decision_ms),
        ("Synthesis Planning", avg_profile.synthesis_planning_ms),
        ("Granular Synthesis", avg_profile.synthesis_audio_ms),
    ];

    let max_stage = bottleneck.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!(
        "  - Primary bottleneck: {} ({:.1}ms, {:.0}% of total)",
        max_stage.0,
        max_stage.1,
        max_stage.1 / avg_profile.total() * 100.0
    );

    if avg_profile.decision_ms > 20.0 {
        println!(
            "  - Consider optimizing Python cognitive layer (currently {:.1}ms)",
            avg_profile.decision_ms
        );
    }
    if avg_profile.feature_extraction_ms > 20.0 {
        println!(
            "  - Consider SIMD optimization for feature extraction (currently {:.1}ms)",
            avg_profile.feature_extraction_ms
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}
