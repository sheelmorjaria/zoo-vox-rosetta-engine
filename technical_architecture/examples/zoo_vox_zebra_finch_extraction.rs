//! Zebra Finch Phrase Extraction - Within-Call Acoustic Similarity Analysis
//!
//! Demonstrates phrase extraction for zebra finches using the acoustic
//! similarity engine for within-call linguistics.
//!
//! Zebra Finch Characteristics:
//! - Song composed of stereotyped "syllables" organized into "motifs"
//! - Combinatorial encoding: discrete syllable types
//! - Frequency range: 2-8 kHz typical
//! - Syllable duration: 50-200ms
//! - Rapid frequency modulations (FM sweeps)
//!
//! Usage:
//!   cargo run --release --example zoo_vox_zebra_finch_extraction

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use technical_architecture::species::SpeciesConfigFactory;
use technical_architecture::{
    AcousticFeatures30D, PhrasePrototype, SimilarityBasedLibraryBuilder, WithinCallAnalyzer, WithinCallConfig,
    ZooVoxExtractionConfig, ZooVoxFeatureExtractor, ZooVoxPhraseExtractor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Zebra Finch Phrase Extraction                              ║");
    println!("║     Within-Call Acoustic Similarity Analysis                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let sample_rate: u32 = 48000;

    // ========================================================================
    // Step 1: Zebra Finch Species Configuration
    // ========================================================================
    println!("Step 1: Zebra Finch Species Configuration\n");
    println!("─────────────────────────────────────────");

    let config = SpeciesConfigFactory::create("zebra_finch");

    println!("  Species:           {}", config.species());
    println!("  Encoding Strategy: {:?}", config.encoding_strategy());
    println!("  Analysis Modality: {:?}", config.modality());

    let params = config.feature_params();
    println!("\n  Feature Parameters:");
    println!(
        "    Phrase duration: {:.0} - {:.0} ms",
        params.phrase_min_ms, params.phrase_max_ms
    );
    println!("    Similarity threshold: {:.2}", params.similarity_threshold);
    println!("    Feature dimensions: {}", params.feature_dim);

    let contexts = config.context_labels();
    println!("\n  Context Labels ({}):", contexts.len());
    for (i, ctx) in contexts.iter().take(8).enumerate() {
        println!("    {}. {}", i + 1, ctx);
    }
    if contexts.len() > 8 {
        println!("    ... and {} more", contexts.len() - 8);
    }
    println!();

    // ========================================================================
    // Step 2: Generate Synthetic Zebra Finch Song
    // ========================================================================
    println!("Step 2: Generating Synthetic Zebra Finch Song\n");
    println!("─────────────────────────────────────────────");

    // Zebra finch songs have characteristic syllable types:
    // - Introduction notes (high frequency, short)
    // - Stack syllables (rapid FM)
    // - Distance calls (longer, harmonic)
    // - Motif patterns (repeated syllable sequences)

    let syllable_specs = [
        ("Intro", 4500.0, 50.0, 0.05, "high_short"),   // Introduction note
        ("Stack_A", 3500.0, 100.0, 0.1, "fm_rapid"),   // Rapid FM sweep
        ("Stack_B", 2800.0, 80.0, 0.08, "fm_rapid"),   // Another FM sweep
        ("Distance", 3200.0, 180.0, 0.18, "harmonic"), // Distance call
        ("Motif_1", 4000.0, 60.0, 0.06, "trill"),      // Trill syllable
        ("Motif_2", 3800.0, 70.0, 0.07, "trill"),      // Another trill
    ];

    println!("  Syllable Types in Generated Song:");
    for (name, freq, dur_ms, dur_sec, style) in &syllable_specs {
        println!("    {} - {:.0}Hz, {:.0}ms ({})", name, freq, dur_ms, style);
    }

    // Generate a motif sequence typical of zebra finch song
    // Pattern: Intro → Stack_A → Stack_B → Motif_1 → Motif_2 (repeated)
    let motif_pattern = [0, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5]; // Syllable indices
    let gap_duration = 0.04; // 40ms gaps between syllables

    // Calculate total duration
    let total_syllable_duration: f64 = motif_pattern.iter().map(|&i| syllable_specs[i].3).sum::<f64>();
    let total_gap_duration = gap_duration * (motif_pattern.len() - 1) as f64;
    let total_duration = total_syllable_duration + total_gap_duration;

    println!("\n  Motif Pattern: {} syllables", motif_pattern.len());
    println!("  Total duration: {:.2}s", total_duration);
    println!();

    // Generate the audio
    let total_samples = (sample_rate as f64 * (total_duration + 0.2)) as usize;
    let mut audio: Vec<f64> = vec![0.0; total_samples];
    let mut current_time = 0.1; // Start with 100ms silence

    for &syllable_idx in &motif_pattern {
        let (name, freq, _dur_ms, dur_sec, style) = &syllable_specs[syllable_idx];
        let start_sample = (current_time * sample_rate as f64) as usize;
        let syllable_samples = (*dur_sec * sample_rate as f64) as usize;

        for i in 0..syllable_samples {
            let sample_idx = start_sample + i;
            if sample_idx >= total_samples {
                break;
            }

            let t = i as f64 / sample_rate as f64;
            let global_t = current_time + t;

            let sample = match *style {
                "high_short" => {
                    // Short high-frequency note
                    let carrier = (2.0 * std::f64::consts::PI * freq * global_t).sin();
                    let envelope = (1.0 - (t / dur_sec).powi(2)).max(0.0);
                    carrier * envelope * 0.4
                }
                "fm_rapid" => {
                    // Rapid frequency modulation
                    let fm = 200.0 * (2.0 * std::f64::consts::PI * 40.0 * t).sin();
                    let carrier = (2.0 * std::f64::consts::PI * (freq + fm) * global_t).sin();
                    let envelope = ((t / dur_sec) * std::f64::consts::PI).sin();
                    carrier * envelope * 0.5
                }
                "harmonic" => {
                    // Harmonic call with multiple partials
                    let f0 = (2.0 * std::f64::consts::PI * freq * global_t).sin();
                    let h2 = 0.3 * (2.0 * std::f64::consts::PI * freq * 2.0 * global_t).sin();
                    let h3 = 0.15 * (2.0 * std::f64::consts::PI * freq * 3.0 * global_t).sin();
                    let envelope = ((t / dur_sec) * std::f64::consts::PI).sin();
                    (f0 + h2 + h3) * envelope * 0.4
                }
                "trill" => {
                    // Trill with amplitude modulation
                    let carrier = (2.0 * std::f64::consts::PI * freq * global_t).sin();
                    let am = 0.5 * (2.0 * std::f64::consts::PI * 80.0 * t).sin() + 0.5;
                    let envelope = ((t / dur_sec) * std::f64::consts::PI).sin();
                    carrier * am * envelope * 0.5
                }
                _ => 0.0,
            };

            audio[sample_idx] = sample;
        }

        current_time += dur_sec + gap_duration;
    }

    println!(
        "  Generated audio: {} samples ({:.2}s)",
        audio.len(),
        audio.len() as f64 / sample_rate as f64
    );
    println!();

    // ========================================================================
    // Step 3: Extract Phrases with Zebra Finch-Specific Segmentation
    // ========================================================================
    println!("Step 3: Extracting Phrases with Species-Specific Segmentation\n");
    println!("─────────────────────────────────────────────────────────────────");

    let extraction_config = ZooVoxExtractionConfig::for_species("zebra_finch", sample_rate);
    let mut phrase_extractor = ZooVoxPhraseExtractor::new(extraction_config);

    let phrases = phrase_extractor.extract_phrases(&audio, "zebra_finch", None)?;

    println!("  Extracted {} phrase candidates", phrases.len());
    println!();

    // Display extracted phrases
    println!("  Phrase Candidates (first 10):");
    println!(
        "  {:<4} {:<12} {:<10} {:<10} {:<15}",
        "#", "Key", "F0 (Hz)", "Dur (ms)", "HNR (dB)"
    );
    println!("  {}", "-".repeat(55));

    for (i, phrase) in phrases.iter().take(10).enumerate() {
        println!(
            "  {:<4} {:<12} {:<10.0} {:<10.0} {:<15.1}",
            i + 1,
            phrase.phrase_key.split("_").take(3).collect::<Vec<_>>().join("_"),
            phrase.features_30d.mean_f0_hz,
            phrase.features_30d.duration_ms,
            phrase.features_30d.harmonic_to_noise_ratio
        );
    }
    if phrases.len() > 10 {
        println!("  ... and {} more", phrases.len() - 10);
    }
    println!();

    // ========================================================================
    // Step 4: Within-Call Analysis Using Acoustic Similarity
    // ========================================================================
    println!("Step 4: Within-Call Analysis Using Acoustic Similarity\n");
    println!("───────────────────────────────────────────────────────");

    // Use zebra finch-specific configuration
    let mut analyzer = WithinCallAnalyzer::for_species("zebra_finch");
    let result = analyzer.discover_phrases(phrases.clone(), "zf_song_001", "zebra_finch");

    println!("  === Phrase Type Discovery ===");
    println!("  Total phrases:     {}", result.total_phrases);
    println!("  Unique types:      {}", result.unique_types);
    println!("  Type entropy:      {:.3} bits", result.type_entropy);
    println!();

    // Display discovered phrase types
    println!("  Discovered Phrase Types:");
    println!(
        "  {:<4} {:<20} {:>8} {:>10} {:>10} {:>12}",
        "#", "Type ID", "Count", "F0 (Hz)", "Dur (ms)", "Variability"
    );
    println!("  {}", "-".repeat(70));

    for (i, pt) in result.phrase_types.iter().enumerate() {
        println!(
            "  {:<4} {:<20} {:>8} {:>10.0} {:>10.0} {:>12.4}",
            i + 1,
            pt.type_id.split('_').last().unwrap_or("?"),
            pt.occurrence_count,
            pt.centroid_features.mean_f0_hz,
            pt.centroid_features.duration_ms,
            pt.intra_variability
        );
    }
    println!();

    // ========================================================================
    // Step 5: Similarity Statistics
    // ========================================================================
    println!("Step 5: Acoustic Similarity Statistics\n");
    println!("──────────────────────────────────────");

    println!(
        "  Within-type similarity:  {:.4} (higher = more cohesive)",
        result.avg_within_type_similarity
    );
    println!(
        "  Between-type distance:   {:.4} (higher = more separated)",
        result.avg_between_type_distance
    );

    let separation_ratio = if result.avg_within_type_similarity > 0.0 && result.avg_within_type_similarity < 1.0 {
        result.avg_between_type_distance / (1.0 - result.avg_within_type_similarity)
    } else {
        f64::INFINITY
    };
    println!(
        "  Separation ratio:        {:.2}x (higher = better discrimination)",
        separation_ratio
    );
    println!();

    // ========================================================================
    // Step 6: Phrase Sequence Analysis
    // ========================================================================
    println!("Step 6: Phrase Sequence Analysis\n");
    println!("────────────────────────────────");

    println!("  Phrase Sequence ({} elements):", result.phrase_sequence.len());

    // Group into lines of 8
    for chunk in result.phrase_sequence.chunks(8) {
        let line: Vec<&str> = chunk.iter().map(|s| s.split('_').last().unwrap_or("?")).collect();
        println!("    {}", line.join(" → "));
    }
    println!();

    // Transition analysis
    println!("  Transition Matrix (top transitions):");
    let mut transitions: Vec<_> = result
        .transition_matrix
        .iter()
        .flat_map(|(from, inner)| inner.iter().map(move |(to, count)| (from.clone(), to.clone(), *count)))
        .collect();
    transitions.sort_by(|a, b| b.2.cmp(&a.2));

    for (from, to, count) in transitions.iter().take(8) {
        let from_short = from.split('_').last().unwrap_or("?");
        let to_short = to.split('_').last().unwrap_or("?");
        println!("    type_{} → type_{}: {} occurrences", from_short, to_short, count);
    }
    println!();

    // ========================================================================
    // Step 7: Motif Discovery
    // ========================================================================
    println!("Step 7: Motif Discovery\n");
    println!("───────────────────────");

    // Find motifs with minimum length 2 and at least 2 occurrences
    let motifs = analyzer.find_motifs(&result, 2, 2);

    if motifs.is_empty() {
        println!("  No recurring motifs found (need more repetitions)");
        println!("  Hint: In real zebra finch data, motifs repeat frequently");
    } else {
        println!("  Discovered {} recurring motif(s):\n", motifs.len());

        for (i, motif) in motifs.iter().take(5).enumerate() {
            let pattern: Vec<&str> = motif
                .pattern
                .iter()
                .map(|s| s.split('_').last().unwrap_or("?"))
                .collect();

            println!("  Motif {}:", i + 1);
            println!("    Pattern: [{}]", pattern.join(", "));
            println!("    Occurrences: {}", motif.occurrence_count);
            println!("    Positions: {:?}", motif.positions);
            println!();
        }
    }
    println!();

    // ========================================================================
    // Step 8: Feature Analysis for Zebra Finch
    // ========================================================================
    println!("Step 8: Feature Analysis for Zebra Finch Phrases\n");
    println!("─────────────────────────────────────────────────");

    if !result.phrase_types.is_empty() {
        // Analyze feature importance for discrimination
        println!("  Representative Phrase Type Features:\n");

        let pt = &result.phrase_types[0];
        let f = &pt.centroid_features;

        println!("  === FUNDAMENTAL (3 features) ===");
        println!("    Mean F0:        {:>8.1} Hz  (typical: 2000-8000 Hz)", f.mean_f0_hz);
        println!("    Duration:       {:>8.1} ms   (typical: 50-200 ms)", f.duration_ms);
        println!("    F0 Range:       {:>8.1} Hz", f.f0_range_hz);

        println!("\n  === GRIT FACTORS (3 features) ===");
        println!(
            "    HNR:            {:>8.1} dB   (harmonic calls have higher HNR)",
            f.harmonic_to_noise_ratio
        );
        println!(
            "    Spectral Flat:  {:>8.3}     (lower = more tonal)",
            f.spectral_flatness
        );
        println!("    Harmonicity:    {:>8.3}", f.harmonicity);

        println!("\n  === MOTION FACTORS (7 features) ===");
        println!(
            "    Attack Time:    {:>8.1} ms   (FM sweeps have faster attack)",
            f.attack_time_ms
        );
        println!("    Decay Time:     {:>8.1} ms", f.decay_time_ms);
        println!("    Sustain Level:  {:>8.3}", f.sustain_level);
        println!(
            "    Vibrato Rate:   {:>8.1} Hz  (trills have high rate ~80 Hz)",
            f.vibrato_rate_hz
        );
        println!("    Vibrato Depth:  {:>8.2} st", f.vibrato_depth);

        println!("\n  === FINGERPRINT FACTORS (14 features) ===");
        println!(
            "    MFCC 1-3:       [{:.1}, {:.1}, {:.1}]",
            f.mfcc_1, f.mfcc_2, f.mfcc_3
        );
        println!("    Spectral Flux:  {:>8.1}", f.spectral_flux);

        println!("\n  === RHYTHM FACTORS (3 features) ===");
        println!("    Median ICI:     {:>8.1} ms  (inter-call interval)", f.median_ici_ms);
        println!("    Onset Rate:     {:>8.1} Hz", f.onset_rate_hz);
    }
    println!();

    // ========================================================================
    // Step 9: Build Zebra Finch Phrase Library
    // ========================================================================
    println!("Step 9: Building Zebra Finch Phrase Library\n");
    println!("────────────────────────────────────────────");

    let library_builder = SimilarityBasedLibraryBuilder::for_species("zebra_finch");
    let library = library_builder.build_library(phrases, "zebra_finch")?;

    println!("  === Zebra Finch Phrase Library ===");
    println!("  Species:             {}", library.species);
    println!("  Encoding Strategy:   {:?}", library.encoding_strategy);
    println!("  Encoding Modality:   {:?}", library.encoding_modality);
    println!("  Total Phrase Types:  {}", library.total_phrases);
    println!("  Total Occurrences:   {}", library.total_occurrences);
    println!("  Type Entropy:        {:.3} bits", library.type_entropy);
    println!(
        "  Frequency Range:     {:.0} - {:.0} Hz",
        library.frequency_range_hz.0, library.frequency_range_hz.1
    );
    println!(
        "  Duration Range:      {:.0} - {:.0} ms",
        library.typical_duration_ms.0, library.typical_duration_ms.1
    );
    println!("  Context Labels:      {}", library.context_labels.len());
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Analysis Summary                            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("✓ Zebra finch-specific configuration applied");
    println!("✓ Combinatorial encoding strategy used");
    println!("✓ Acoustic similarity-based phrase typing");
    println!("✓ Motif discovery for stereotyped sequences");
    println!("✓ Full 30D feature extraction");
    println!("✓ Phrase library generation");
    println!();
    println!("Zebra Finch Characteristics Captured:");
    println!("  • Syllable-based song organization");
    println!("  • Rapid FM sweeps in stack syllables");
    println!("  • Harmonic structure in distance calls");
    println!("  • Trill patterns in motif syllables");
    println!("  • Typical frequency range: 2-8 kHz");

    Ok(())
}
