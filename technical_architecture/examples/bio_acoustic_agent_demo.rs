//! Bio-Acoustic Interaction Agent Demonstration
//!
//! Demonstrates the complete interaction loop:
//! 1. Listen: RosettaPipeline processes audio → ContextEnrichedPhrase
//! 2. Decide: Logic Layer selects response strategy
//! 3. Select: Router retrieves prototype from AcousticInventory
//! 4. Calculate: ContextDeltaCalculator converts environment → Delta
//! 5. Check: Validate Delta doesn't violate Formant Barrier
//! 6. Synthesize: GranularConcatenativeSynthesizer applies Delta
//! 7. Speak: Output valid, context-aware response

use technical_architecture::{
    bio_acoustic_agent::{
        BioAcousticAgent, AcousticInventory, AcousticPrototype, AcousticModality,
        SourceMetadata, SynthesisRequest, EnvState, InteractionContext,
        ContextDeltaCalculator, FormantBarrierValidator,
    },
    rosetta_pipeline::ContextEnrichedPhrase,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Bio-Acoustic Interaction Agent Demonstration                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // STEP 1: Create Acoustic Inventory with Prototypes
    // =========================================================================
    println!("[1/6] Creating Acoustic Inventory with prototypes...");
    println!();

    let mut inventory = AcousticInventory::new("marmoset");

    // Phee - contact call (harmonic)
    inventory.add_prototype(AcousticPrototype {
        label: "Phee".to_string(),
        audio_buffer: vec![0.1; 4800], // 100ms at 48kHz (placeholder)
        sample_rate: 48000,
        metadata: SourceMetadata {
            mean_f0_hz: 7000.0,
            duration_ms: 300.0,
            f0_range_hz: 500.0,
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

    // Tsik - alarm call (transient)
    inventory.add_prototype(AcousticPrototype {
        label: "Tsik".to_string(),
        audio_buffer: vec![0.2; 1200], // 25ms at 48kHz (placeholder)
        sample_rate: 48000,
        metadata: SourceMetadata {
            mean_f0_hz: 9000.0,
            duration_ms: 80.0,
            f0_range_hz: 200.0,
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

    // Twitter - social bonding (mixed)
    inventory.add_prototype(AcousticPrototype {
        label: "Twitter".to_string(),
        audio_buffer: vec![0.15; 2400], // 50ms at 48kHz (placeholder)
        sample_rate: 48000,
        metadata: SourceMetadata {
            mean_f0_hz: 8000.0,
            duration_ms: 200.0,
            f0_range_hz: 1500.0,
            harmonic_to_noise_ratio: 15.0,
            entropy: 0.3,
            attack_time_ms: 10.0,
            sustain_level: 0.5,
            jitter: 0.05,
            loudness: 0.5,
            ..Default::default()
        },
        sample_count: 75,
        modality: AcousticModality::Mixed,
    });

    // Set response strategies
    inventory.set_response_strategy("Tsik", "Phee");  // Calm alarm with contact
    inventory.set_response_strategy("Phee", "Phee");  // Reply to contact
    inventory.set_response_strategy("Twitter", "Twitter"); // Echo social

    println!("  Added prototypes:");
    for label in inventory.available_labels() {
        let proto = inventory.get_prototype(label).unwrap();
        println!("    ├─ {}: F0={:.0}Hz, Duration={:.0}ms, Modality={:?}",
            label, proto.metadata.mean_f0_hz, proto.metadata.duration_ms, proto.modality);
    }
    println!();

    // =========================================================================
    // STEP 2: Create Bio-Acoustic Agent
    // =========================================================================
    println!("[2/6] Creating Bio-Acoustic Interaction Agent...");
    println!();

    let agent = BioAcousticAgent::new(inventory, 48000);

    // =========================================================================
    // STEP 3: Demonstrate Context-to-Delta Mapping (Acoustic Algebra)
    // =========================================================================
    println!("[3/6] Demonstrating Context-to-Delta Mapping (Acoustic Algebra)...");
    println!();

    println!("  Environmental Adaptations:");
    println!("  ┌─────────────┬─────────────────────────────────────────────────────────────┐");
    println!("  │ Environment │ Delta Transformation                                          │");
    println!("  ├─────────────┼─────────────────────────────────────────────────────────────┤");

    let env_deltas = vec![
        (EnvState::Quiet, "No adaptation"),
        (EnvState::Wind, "Long_Range_Contact"),
        (EnvState::Rain, "Moderate boost"),
        (EnvState::Storm, "Emergency signal"),
    ];

    for (env, effect) in env_deltas {
        let delta = ContextDeltaCalculator::calculate(env, InteractionContext::Solo);
        println!("  │ {:?} │ +{:.0}Hz, +{:.2} loudness ({})",
            env, delta.delta_mean_f0_hz, delta.delta_loudness, effect);
    }
    println!("  └─────────────┴─────────────────────────────────────────────────────────────┘");
    println!();

    // =========================================================================
    // STEP 4: Demonstrate Synthesis Planning
    // =========================================================================
    println!("[4/6] Demonstrating Synthesis Planning...");
    println!();

    // Test 1: Phee in wind
    println!("  Test 1: Synthesize 'Phee' in windy conditions");
    let request = SynthesisRequest::new("Phee")
        .with_environment(EnvState::Wind)
        .with_context(InteractionContext::Reply);
    let plan = agent.plan_synthesis(request)?;
    println!("    ├─ Source: F0={:.0}Hz", plan.source_metadata.mean_f0_hz);
    println!("    ├─ Delta: +{:.0}Hz pitch, +{:.2} loudness", plan.delta.delta_mean_f0_hz, plan.delta.delta_loudness);
    println!("    ├─ Target: F0={:.0}Hz", plan.target_metadata.mean_f0_hz);
    println!("    └─ Valid: {} ✓", plan.validation.is_valid);
    println!();

    // Test 2: Tsik with high emotional intensity
    println!("  Test 2: Synthesize 'Tsik' with high urgency (grading=0.9)");
    let request = SynthesisRequest::new("Tsik")
        .with_grading(0.9);
    let plan = agent.plan_synthesis(request)?;
    println!("    ├─ Source: jitter={:.2}", plan.source_metadata.jitter);
    println!("    ├─ Delta: +{:.2} jitter, +{:.2} shimmer", plan.delta.delta_jitter, plan.delta.delta_shimmer);
    println!("    ├─ Target: jitter={:.2}", plan.target_metadata.jitter);
    println!("    └─ Valid: {} ✓", plan.validation.is_valid);
    println!();

    // Test 3: Combined deltas
    println!("  Test 3: Combined deltas (Wind + Reply + Grading)");
    let env_delta = ContextDeltaCalculator::calculate(EnvState::Wind, InteractionContext::Reply);
    let grading_delta = ContextDeltaCalculator::calculate_for_grading(0.8);
    let combined = ContextDeltaCalculator::combine(&[env_delta, grading_delta]);
    println!("    └─ Combined: +{:.0}Hz pitch, +{:.2} jitter", combined.delta_mean_f0_hz, combined.delta_jitter);
    println!();

    // =========================================================================
    // STEP 5: Demonstrate Formant Barrier Validation
    // =========================================================================
    println!("[5/6] Demonstrating Formant Barrier Validation...");
    println!();

    // Valid transformation
    println!("  Test 1: Valid transformation (Harmonic → Harmonic)");
    let source = SourceMetadata {
        harmonic_to_noise_ratio: 20.0,
        entropy: 0.15,
        ..Default::default()
    };
    let target = SourceMetadata {
        harmonic_to_noise_ratio: 25.0,
        entropy: 0.20,
        ..Default::default()
    };
    let validation = FormantBarrierValidator::validate(&source, &target);
    println!("    ├─ Source: HNR={:.0}, Entropy={:.2}", source.harmonic_to_noise_ratio, source.entropy);
    println!("    ├─ Target: HNR={:.0}, Entropy={:.2}", target.harmonic_to_noise_ratio, target.entropy);
    println!("    └─ Valid: {} ✓", validation.is_valid);
    println!();

    // Invalid transformation (crossing barrier)
    println!("  Test 2: Invalid transformation (Harmonic → Transient)");
    let source = SourceMetadata {
        harmonic_to_noise_ratio: 25.0,
        entropy: 0.1,
        duration_ms: 300.0,
        ..Default::default()
    };
    let target = SourceMetadata {
        harmonic_to_noise_ratio: 5.0,
        entropy: 0.8,
        duration_ms: 50.0,
        ..Default::default()
    };
    let validation = FormantBarrierValidator::validate(&source, &target);
    println!("    ├─ Source: HNR={:.0}, Entropy={:.2} (Harmonic)", source.harmonic_to_noise_ratio, source.entropy);
    println!("    ├─ Target: HNR={:.0}, Entropy={:.2} (Transient)", target.harmonic_to_noise_ratio, target.entropy);
    println!("    ├─ Valid: {} ✗", validation.is_valid);
    println!("    ├─ Violations:");
    for v in &validation.violations {
        println!("    │   └─ {}", v);
    }
    println!("    └─ Action: {}", validation.recommended_action);
    println!();

    // =========================================================================
    // STEP 6: Complete Interaction Loop
    // =========================================================================
    println!("[6/6] Complete Interaction Loop...");
    println!();

    // Simulate input phrase from RosettaPipeline
    let input_phrase = ContextEnrichedPhrase {
        phrase_type_id: "Type_52".to_string(),
        grading_score: 0.7,
        acoustic_confidence: 0.85,
        semantic_label: "Tsik".to_string(),
        label_confidence: 0.9,
        syntax_role: technical_architecture::rosetta_pipeline::SyntaxRole::Initiator,
        environmental_state: technical_architecture::rosetta_pipeline::EnvState::Wind,
        inferred_intent: "Warning".to_string(),
        start_ms: 0.0,
        duration_ms: 80.0,
    };

    println!("  Input Phrase (from RosettaPipeline):");
    println!("    ├─ Semantic Label: {} ({:.0}% confidence)", input_phrase.semantic_label, input_phrase.label_confidence * 100.0);
    println!("    ├─ Inferred Intent: {}", input_phrase.inferred_intent);
    println!("    ├─ Environment: {:?}", input_phrase.environmental_state);
    println!("    └─ Grading Score: {:.1}", input_phrase.grading_score);
    println!();

    // Select response
    let response_label = agent.select_response(&input_phrase.semantic_label)
        .map(|s| s.clone())
        .unwrap_or_else(|| input_phrase.semantic_label.clone());

    println!("  Response Strategy:");
    println!("    └─ Input '{}' → Response '{}'", input_phrase.semantic_label, response_label);
    println!();

    // Plan synthesis
    let env = match input_phrase.environmental_state {
        technical_architecture::rosetta_pipeline::EnvState::Quiet => EnvState::Quiet,
        technical_architecture::rosetta_pipeline::EnvState::Wind => EnvState::Wind,
        technical_architecture::rosetta_pipeline::EnvState::Rain => EnvState::Rain,
        technical_architecture::rosetta_pipeline::EnvState::Storm => EnvState::Storm,
        _ => EnvState::Unknown,
    };

    let request = SynthesisRequest::new(&response_label)
        .with_environment(env)
        .with_context(InteractionContext::Reply)
        .with_grading(input_phrase.grading_score * 0.5); // Reduce intensity in reply

    let plan = agent.plan_synthesis(request)?;

    println!("  Synthesis Plan:");
    println!("    ├─ Source: '{}' prototype", plan.source_label);
    println!("    ├─ F0: {:.0}Hz → {:.0}Hz (+{:.0}Hz)",
        plan.source_metadata.mean_f0_hz,
        plan.target_metadata.mean_f0_hz,
        plan.delta.delta_mean_f0_hz);
    println!("    ├─ Loudness: {:.2} → {:.2}",
        plan.source_metadata.loudness,
        plan.target_metadata.loudness);
    println!("    ├─ Jitter: {:.2} → {:.2}",
        plan.source_metadata.jitter,
        plan.target_metadata.jitter);
    println!("    └─ Valid: {} ✓", plan.validation.is_valid);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("The Bio-Acoustic Interaction Agent successfully:");
    println!();
    println!("  1. LISTENED: Processed ContextEnrichedPhrase from RosettaPipeline");
    println!("  2. DECIDED: Selected '{}' as response to '{}'", response_label, input_phrase.semantic_label);
    println!("  3. SELECTED: Retrieved prototype from AcousticInventory");
    println!("  4. CALCULATED: Generated context-aware deltas (Wind + Reply + Grading)");
    println!("  5. CHECKED: Validated against Formant Barrier");
    println!("  6. SYNTHESIZED: Ready for output (plan created)");
    println!();
    println!("Next step: Pass SynthesisPlan to GranularConcatenativeSynthesizer");
    println!("for audio generation.");

    Ok(())
}
