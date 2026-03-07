// Linguistic Analysis Demo: Marmoset Vocalizations
//
// This example demonstrates comprehensive linguistic analysis on marmoset vocalization data,
// including:
// - Information Theory (Zipf's Law)
// - Prosody (Isochrony/Rhythm)
// - Phonotactics (Forbidden Transitions)
// - Pragmatics (Turn-Taking)
// - Updated Atomicity with Usage Frequency
//
// Usage: cargo run --example linguistic_analysis_marmoset

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use technical_architecture::{
    ClusteredPhrase, ExtractionConfig, ExtractionPhraseCandidate as PhraseCandidate, LinguisticAnalysis,
    ParallelExtractionPipeline, VocalizationResult,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Linguistic Analysis Demo: Marmoset Vocalization Dataset              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create pipeline with default configuration
    let config = ExtractionConfig::default();
    let pipeline = ParallelExtractionPipeline::with_config(config)?;

    println!("✓ Parallel extraction pipeline created");
    println!("  - Workers: {}", pipeline.config().num_workers);
    println!("  - Sample rate: {} Hz", pipeline.config().sample_rate);
    println!();

    // ========================================================================
    // Step 1: Load or simulate marmoset vocalization data
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Marmoset Vocalization Data                                │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // For demonstration, we'll create synthetic marmoset data based on real characteristics
    // In production, you would load actual audio files and process them
    let (vocalization_results, clustered_phrases) = create_marmoset_demo_data()?;

    println!("✓ Loaded marmoset vocalization data:");
    println!("  - Vocalizations: {}", vocalization_results.len());
    println!(
        "  - Total phrases: {}",
        vocalization_results.iter().map(|v| v.phrases.len()).sum::<usize>()
    );
    println!("  - Clustered phrases: {}", clustered_phrases.len());
    println!();

    // ========================================================================
    // Step 2: Run Comprehensive Linguistic Analysis
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Running Linguistic Analysis                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let analysis = pipeline.analyze_linguistics(&vocalization_results, &clustered_phrases)?;

    println!("✓ Linguistic analysis complete");
    println!();

    // ========================================================================
    // Step 3: Display Results - Information Theory (Zipf's Law)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. Information Theory: Zipf's Law (Least Effort Principle)              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("Zipf's Law Analysis:");
    println!("  Slope (α): {:.3}", analysis.zipf.slope_alpha);
    println!("  Correlation (R²): {:.3}", analysis.zipf.correlation_r2);
    println!("  Efficiency: {:?}", analysis.zipf.efficiency);
    println!();

    // Interpret efficiency
    match analysis.zipf.efficiency {
        technical_architecture::CommunicationEfficiency::Optimal { slope } => {
            println!("  → INTERPRETATION: Optimal communication (human-like efficiency)");
            println!("     Marmosets follow Zipf's Law with slope ≈ -1.0, indicating");
            println!("     efficient coding of information in vocalizations.");
        }
        technical_architecture::CommunicationEfficiency::Efficient { slope } => {
            println!("  → INTERPRETATION: Efficient communication");
            println!("     Marmosets show good efficiency, similar to highly social");
            println!("     species with complex vocal repertoires.");
        }
        technical_architecture::CommunicationEfficiency::Inefficient { slope } => {
            println!("  → INTERPRETATION: Less efficient communication");
            println!("     High repetition of phrases, possibly due to limited");
            println!("     vocabulary or specific social context.");
        }
        technical_architecture::CommunicationEfficiency::Random { slope } => {
            println!("  → INTERPRETATION: No clear pattern (uniform distribution)");
            println!("     Phrases appear randomly, suggesting no grammar");
            println!("     or communicative structure.");
        }
        technical_architecture::CommunicationEfficiency::Unknown => {
            println!("  → INTERPRETATION: Insufficient data for classification");
        }
    }
    println!();

    // Top 5 most frequent phrases
    println!("  Top 5 Most Frequent Phrases:");
    for (i, phrase_id) in analysis.zipf.ranked_phrases.iter().take(5).enumerate() {
        let freq = analysis.zipf.phrase_frequencies.get(phrase_id).unwrap_or(&0);
        let rank = i + 1;
        println!("    {}. {} (occurrences: {})", rank, phrase_id, freq);
    }
    println!();

    // ========================================================================
    // Step 4: Display Results - Prosody (Isochrony/Rhythm)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. Prosody: Isochrony (Rhythm Detection)                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("Prosody Analysis:");
    println!("  Gap CV (Coefficient of Variation): {:.3}", analysis.prosody.gap_cv);
    println!("  Mean gap duration: {:.2} ms", analysis.prosody.mean_gap_ms);
    println!("  Gap std deviation: {:.2} ms", analysis.prosody.gap_std_ms);
    println!("  Rhythm classification: {:?}", analysis.prosody.rhythm);
    println!();

    // Interpret rhythm
    match analysis.prosody.rhythm {
        technical_architecture::Rhythmicity::Isochronous { cv } => {
            println!("  → INTERPRETATION: Highly rhythmic (Isochronous)");
            println!("     Marmosets produce vocalizations with metronome-like precision,");
            println!("     suggesting evolved turn-taking mechanisms and social coordination.");
            println!("     This is typical of social contact calls (phee calls).");
        }
        technical_architecture::Rhythmicity::Rhythmic { cv } => {
            println!("  → INTERPRETATION: Moderately rhythmic");
            println!("     Marmosets show consistent timing patterns, though with some");
            println!("     variation. Indicates structured communication.");
        }
        technical_architecture::Rhythmicity::Variable { cv } => {
            println!("  → INTERPRETATION: Variable rhythm");
            println!("     Timing varies significantly, possibly due to context-dependent");
            println!("     vocalizations (e.g., food calls vs. alarm calls).");
        }
        technical_architecture::Rhythmicity::Arrhythmic { cv } => {
            println!("  → INTERPRETATION: Arrhythmic (Staccato/Chaotic)");
            println!("     No consistent rhythm, possibly indicating excitement, alarm,");
            println!("     or competitive vocal exchanges.");
        }
        technical_architecture::Rhythmicity::Unknown => {
            println!("  → INTERPRETATION: Insufficient data for rhythm classification");
        }
    }
    println!();

    // ========================================================================
    // Step 5: Display Results - Phonotactics (Forbidden Transitions)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. Phonotactics: Forbidden Transitions                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("Phonotactics Analysis:");
    println!(
        "  Total unique transitions: {}",
        analysis.phonotactics.transition_matrix.len()
    );
    println!(
        "  Forbidden/rare transitions: {}",
        analysis.phonotactics.forbidden_transitions.len()
    );
    println!(
        "  Mean spectral delta: {:.3}",
        analysis.phonotactics.mean_spectral_delta
    );
    println!();

    if !analysis.phonotactics.forbidden_transitions.is_empty() {
        println!("  Sample of Forbidden/Rare Transitions:");
        for (i, ft) in analysis.phonotactics.forbidden_transitions.iter().take(5).enumerate() {
            println!(
                "    {}. {} → {} (prob: {:.3}, reason: {:?})",
                i + 1,
                ft.from_phrase,
                ft.to_phrase,
                ft.probability,
                ft.reason
            );
        }
        println!();

        println!("  → INTERPRETATION: Physical constraints on vocal production");
        println!("     Some phrase combinations are rarely or never used, suggesting");
        println!("     biomechanical constraints or optimization for ease of articulation.");
    } else {
        println!("  → INTERPRETATION: No strongly forbidden transitions detected");
        println!("     Marmosets freely combine phrases, suggesting flexible vocal");
        println!("     production system.");
    }
    println!();

    // ========================================================================
    // Step 6: Display Results - Pragmatics (Turn-Taking)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. Pragmatics: Turn-Taking Patterns                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("Pragmatics Analysis:");
    println!("  Pattern: {:?}", analysis.pragmatics.pattern);
    println!(
        "  Overlap count: {}",
        analysis.pragmatics.overlap_analysis.overlap_count
    );
    println!("  Mean gap: {:.1} ms", analysis.pragmatics.gap_analysis.mean_gap_ms);
    println!();

    match analysis.pragmatics.pattern {
        technical_architecture::TurnTakingPattern::Strict => {
            println!("  → INTERPRETATION: Strict turn-taking");
            println!("     No overlaps, consistent gaps. Typical of marmoset social");
            println!("     communication, suggesting evolved cooperative signaling.");
        }
        technical_architecture::TurnTakingPattern::Flexible => {
            println!("  → INTERPRETATION: Flexible turn-taking");
            println!("     Some overlaps and variable gaps, indicating adaptive");
            println!("     communication based on social context.");
        }
        technical_architecture::TurnTakingPattern::Overlapping => {
            println!("  → INTERPRETATION: High overlap (Rapid-fire)");
            println!("     Frequent simultaneous vocalizations, possibly indicating");
            println!("     competition, excitement, or group chorusing.");
        }
        technical_architecture::TurnTakingPattern::Unknown => {
            println!("  → INTERPRETATION: Pattern analysis requires speaker identification");
            println!("     Full turn-taking analysis needs individual ID tracking.");
        }
    }
    println!();

    // ========================================================================
    // Step 7: Display Results - Updated Atomicity
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 5. Updated Atomicity: Phonological × Semantic                           │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let phonologically_atomic = analysis
        .updated_atomic_phrases
        .iter()
        .filter(|p| p.is_phonologically_atomic)
        .count();
    let semantically_atomic = analysis
        .updated_atomic_phrases
        .iter()
        .filter(|p| p.is_semantically_atomic)
        .count();
    let truly_atomic = analysis
        .updated_atomic_phrases
        .iter()
        .filter(|p| p.is_truly_atomic)
        .count();

    println!("Updated Atomicity Analysis:");
    println!("  Total phrases analyzed: {}", analysis.updated_atomic_phrases.len());
    println!(
        "  Phonologically atomic: {} ({:.1}%)",
        phonologically_atomic,
        phonologically_atomic as f64 / analysis.updated_atomic_phrases.len() as f64 * 100.0
    );
    println!(
        "  Semantically atomic: {} ({:.1}%)",
        semantically_atomic,
        semantically_atomic as f64 / analysis.updated_atomic_phrases.len() as f64 * 100.0
    );
    println!(
        "  Truly atomic (both): {} ({:.1}%)",
        truly_atomic,
        truly_atomic as f64 / analysis.updated_atomic_phrases.len() as f64 * 100.0
    );
    println!();

    // Sample truly atomic phrases
    println!("  Sample of Truly Atomic Phrases:");
    for (i, phrase) in analysis
        .updated_atomic_phrases
        .iter()
        .filter(|p| p.is_truly_atomic)
        .take(5)
        .enumerate()
    {
        println!(
            "    {}. {} (freq: {}, intra_sim: {:.2}, inter_sim: {:.2})",
            i + 1,
            phrase.phrase_id,
            phrase.frequency,
            phrase.intra_cluster_similarity,
            phrase.inter_cluster_similarity
        );
    }
    println!();

    println!("  → INTERPRETATION: Updated atomicity filters noise from vocabulary");
    println!("     By requiring both acoustic coherence AND usage frequency, we");
    println!("     identify the true building blocks of marmoset communication.");
    println!();

    // ========================================================================
    // Summary and Scientific Conclusions
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                           SCIENTIFIC SUMMARY                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Marmoset Communication Profile:");
    println!();
    println!("1. Information Theory:");
    println!("   - Efficiency Level: {:?}", analysis.zipf.efficiency);
    println!("   - Zipf Slope (α): {:.3} (human ≈ -1.0)", analysis.zipf.slope_alpha);
    println!();

    println!("2. Prosody:");
    println!("   - Rhythm Pattern: {:?}", analysis.prosody.rhythm);
    println!("   - Timing Precision: CV = {:.3}", analysis.prosody.gap_cv);
    println!();

    println!("3. Phonotactics:");
    println!(
        "   - Forbidden Transitions: {}",
        analysis.phonotactics.forbidden_transitions.len()
    );
    println!(
        "   - Production Flexibility: {}",
        if analysis.phonotactics.forbidden_transitions.len() < 10 {
            "High"
        } else {
            "Constrained"
        }
    );
    println!();

    println!("4. Pragmatics:");
    println!("   - Turn-Taking: {:?}", analysis.pragmatics.pattern);
    println!();

    println!("5. Atomic Phrases:");
    println!("   - True Vocabulary: {} phrases", truly_atomic);
    println!(
        "   - Compositionality: {:.1}% of phrases are reusable",
        truly_atomic as f64 / analysis.updated_atomic_phrases.len() as f64 * 100.0
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Analysis complete! Results demonstrate:");
    println!("• Quantitative metrics for cross-species comparison");
    println!("• Publication-ready data for evolutionary linguistics");
    println!("• Foundation for comparative communication studies");
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

// ============================================================================
// Demo Data Generation (Synthetic Marmoset Data)
// ============================================================================
//
// In production, you would load actual audio files and process them
// through the pipeline. For this demo, we create synthetic data that
// mimics real marmoset vocalization patterns.
// ============================================================================

fn create_marmoset_demo_data() -> Result<(Vec<VocalizationResult>, Vec<ClusteredPhrase>), Box<dyn std::error::Error>> {
    // Marmoset vocal characteristics (based on real research):
    // - Phee calls: 7-12 kHz F0 range
    // - Duration: 50-200 ms typical
    // - Rhythmic timing in social contexts
    // - Large vocabulary (~1000+ phrase types)

    let mut results = Vec::new();
    let mut clustered_phrases = Vec::new();

    // Create 50 synthetic vocalizations
    for file_id in 0..50 {
        let num_phrases = 5 + (file_id % 10); // 5-14 phrases per vocalization
        let mut phrases = Vec::new();

        let mut current_time = 0.0;
        for phrase_id in 0..num_phrases {
            // Marmoset-like parameters
            let f0 = 7000.0 + ((phrase_id * 100) % 5000) as f64; // 7-12 kHz
            let duration = 50.0 + ((phrase_id * 15) % 150) as f64; // 50-200 ms
            let gap = 80.0 + ((phrase_id * 20) % 80) as f64; // Rhythmic gaps 80-160ms

            let phrase_key = format!("F0_{:.0}_DUR_{:.0}", f0 / 100.0, duration);

            phrases.push(PhraseCandidate {
                phrase_id: phrase_key.clone(),
                file_name: format!("marmoset_{:03}.wav", file_id),
                start_ms: current_time,
                end_ms: current_time + duration,
                duration_ms: duration,
                features: vec![0.0; 30], // Placeholder 30D features
                rms_amplitude: 0.5 + ((phrase_id * 5) % 50) as f64 * 0.01,
                species: "marmoset".to_string(),
                context: format!("context_{}", phrase_id % 5), // 5 contexts
            });

            current_time += duration + gap;
        }

        results.push(VocalizationResult {
            file_name: format!("marmoset_{:03}.wav", file_id),
            species: "marmoset".to_string(),
            sentences: vec![],
            phrases,
        });
    }

    // Create clustered phrases (simulate DBSCAN results)
    // Create a Zipf-like distribution: common phrases repeated many times
    let common_phrases = vec![
        ("F0_70_DUR_50", 30),
        ("F0_75_DUR_65", 25),
        ("F0_80_DUR_80", 20),
        ("F0_85_DUR_95", 15),
        ("F0_90_DUR_110", 12),
        ("F0_95_DUR_125", 10),
        ("F0_100_DUR_140", 8),
        ("F0_105_DUR_155", 6),
    ];

    for (i, (phrase_id, count)) in common_phrases.iter().enumerate() {
        for _ in 0..*count {
            clustered_phrases.push(ClusteredPhrase {
                phrase: PhraseCandidate {
                    phrase_id: phrase_id.to_string(),
                    file_name: format!("marmoset_{:03}.wav", i),
                    start_ms: 0.0,
                    end_ms: 100.0,
                    duration_ms: 100.0,
                    features: vec![0.0; 30],
                    rms_amplitude: 0.5,
                    species: "marmoset".to_string(),
                    context: "contact".to_string(),
                },
                cluster_id: i as i32,
                intra_cluster_similarity: 0.7 + ((i * 3) % 10) as f64 * 0.03, // 0.7-0.97
                inter_cluster_similarity: 0.2 + ((i * 5) % 10) as f64 * 0.05, // 0.2-0.65
                is_atomic: true,
                contexts: vec![1, 2, 3],
            });
        }
    }

    // Add some rare phrases (Hapax Legomena - occur once)
    for i in 0..20 {
        let phrase_id = format!("F0_{:.0}_DUR_{:.0}", 110 + i, 170 + i * 5);
        clustered_phrases.push(ClusteredPhrase {
            phrase: PhraseCandidate {
                phrase_id: phrase_id.clone(),
                file_name: format!("marmoset_{:03}.wav", 50 + i),
                start_ms: 0.0,
                end_ms: 100.0,
                duration_ms: 100.0,
                features: vec![0.0; 30],
                rms_amplitude: 0.5,
                species: "marmoset".to_string(),
                context: "rare".to_string(),
            },
            cluster_id: 100 + i as i32,
            intra_cluster_similarity: 0.8,
            inter_cluster_similarity: 0.3,
            is_atomic: true,
            contexts: vec![4],
        });
    }

    Ok((results, clustered_phrases))
}
