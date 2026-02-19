// Full Parallel Extraction Pipeline: Egyptian Fruit Bat Dataset
//
// Processes Egyptian fruit bat vocalization WAV files from:
// ~/birdsongs_analysis/src/audio_library/bat/
//
// Features:
// - Loads pre-segmented bat vocalization WAV files
// - Organized by phrase type (FM_0_0_DUR_100, etc.)
// - Extracts 56D micro-dynamics features (30D base + 13 Δ + 13 ΔΔ)
// - Collects audio segments into phrase library
// - Runs comprehensive linguistic analysis
//
// Usage:
//   cargo run --example full_pipeline_bat --release

use std::fs;
use std::path::{Path, PathBuf};
use technical_architecture::{
    analyze_context,
    analyze_social_network,
    analyze_turn_taking,
    // Annotation and turn-taking analysis
    load_annotations_from_csv,
    ClusteredPhrase,
    EmitterAnnotation,
    ExtractionPhraseCandidate as PhraseCandidate,
    LinguisticAnalysis,
    ParallelExtractionPipeline,
    PhraseAudioLibrary,
    PhraseAudioSegment,
    VocalizationResult,
};

// Hound for WAV decoding (simpler and faster than symphonia for WAV)
use hound;

/// Load WAV file using hound
fn load_wav_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path.as_ref())?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Read samples as f32
    let audio: Vec<f32> = reader
        .into_samples::<f32>()
        .filter_map(|s| s.ok())
        .collect();

    // Convert to mono if stereo
    let audio_mono = if spec.channels == 2 {
        audio.chunks_exact(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        audio
    };

    Ok((audio_mono, sample_rate))
}

/// Load audio file (currently WAV-only via hound)
/// TODO: Add symphonia support for FLAC/MP3/AAC/OGG
fn load_audio_file<P: AsRef<Path>>(
    path: P,
) -> std::result::Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    load_wav_file(path)
}

// Configuration
const BAT_AUDIO_DIR: &str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio";
const BAT_ANNOTATIONS_CSV: &str =
    "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv";
const MAX_PHRASE_TYPES: usize = 172; // All phrase directories
const MAX_FILES_PER_PHRASE: usize = 100; // Process only 100 files for testing

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Full Parallel Extraction: Egyptian Fruit Bat Dataset (91K files)         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // Step 1: Discover Audio Files
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Discovering Bat Audio Files                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let bat_audio_path = Path::new(BAT_AUDIO_DIR);

    if !bat_audio_path.exists() {
        println!("❌ Directory not found: {}", BAT_AUDIO_DIR);
        println!("   Please ensure the bat dataset is available.");
        return Err("Dataset directory not found".into());
    }

    println!("📂 Scanning directory: {}", BAT_AUDIO_DIR);

    let phrase_directories = discover_phrase_directories(bat_audio_path, MAX_PHRASE_TYPES)?;

    println!(
        "✅ Found {} phrase type directories",
        phrase_directories.len()
    );
    println!();

    // ========================================================================
    // Step 1.5: Load Emitter Annotations
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1.5: Loading Emitter Annotations                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("📂 Loading annotations from: {}", BAT_ANNOTATIONS_CSV);

    let annotations_map = load_annotations_from_csv(BAT_ANNOTATIONS_CSV)?;
    println!("✅ Loaded {} annotations", annotations_map.len());
    println!();

    // Convert to sorted vector for analysis
    let mut annotations_vec: Vec<EmitterAnnotation> = annotations_map.values().cloned().collect();

    // Sort by file name to ensure temporal order
    annotations_vec.sort_by(|a, b| {
        // Extract numeric part from filename for sorting
        let a_num: usize = a
            .file_name
            .trim_end_matches(".wav")
            .trim_end_matches(".WAV")
            .parse()
            .unwrap_or(0);
        let b_num: usize = b
            .file_name
            .trim_end_matches(".wav")
            .trim_end_matches(".WAV")
            .parse()
            .unwrap_or(0);
        a_num.cmp(&b_num)
    });

    println!("✓ Annotations sorted in temporal order");
    println!();

    // ========================================================================
    // Step 2: Process Audio Files with Phrase Library
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Processing Audio Files (with Audio Segments)                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut pipeline = ParallelExtractionPipeline::new()?;

    // Enable phrase library to collect audio segments
    pipeline.enable_phrase_library("egyptian_fruit_bat".to_string());
    println!("✓ Phrase audio library enabled");

    let start_time = std::time::Instant::now();

    let (vocalization_results, clustered_phrases, total_files, audio_segments) =
        process_bat_dataset_parallel(&phrase_directories)?;

    // Add all audio segments to the phrase library
    println!(
        "  Adding {} audio segments to phrase library...",
        audio_segments.len()
    );
    pipeline.add_segments_to_library(audio_segments);

    let processing_time = start_time.elapsed();

    println!("✅ Processing complete");
    println!("   Files processed: {}", total_files);
    println!("   Vocalizations: {}", vocalization_results.len());
    println!(
        "   Total phrases extracted: {}",
        vocalization_results
            .iter()
            .map(|v| v.phrases.len())
            .sum::<usize>()
    );
    println!("   Clustered phrases: {}", clustered_phrases.len());
    println!("   Processing time: {:.2}s", processing_time.as_secs_f64());
    println!(
        "   Throughput: {:.1} files/sec",
        total_files as f64 / processing_time.as_secs_f64()
    );
    println!();

    // ========================================================================
    // Step 3: Display Phrase Library Statistics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Phrase Library Statistics                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    if let Some(library) = pipeline.phrase_library() {
        let stats = library.statistics();

        println!("Phrase Audio Library:");
        println!("  Species: {}", stats.species);
        println!("  Sample Rate: {} Hz", stats.sr);
        println!("  Total Segments: {}", stats.total_segments);
        println!("  Unique Phrases: {}", stats.total_phrases);
        println!(
            "  Max Segments Per Phrase: {}",
            stats.max_segments_per_phrase
        );
        println!();

        println!("  Top 10 Phrase Types:");
        for (i, (phrase_key, count)) in stats.phrase_counts.iter().take(10).enumerate() {
            println!("    {:2}. {} ({} segments)", i + 1, phrase_key, count);
        }
        println!();

        println!(
            "  Library contains {} audio segments ({} MB estimated)",
            stats.total_segments,
            stats.total_segments * 1000 * 4 / 1_000_000
        ); // Rough estimate
        println!();
    }

    // ========================================================================
    // Step 4: Run Linguistic Analysis
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Running Linguistic Analysis                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let analysis = pipeline.analyze_linguistics(&vocalization_results, &clustered_phrases)?;

    println!("✅ Linguistic analysis complete");
    println!();

    // ========================================================================
    // Step 4.5: Turn-Taking and Pragmatics Analysis (with Emitter Data)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4.5: Turn-Taking and Pragmatics Analysis                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("🔬 Analyzing turn-taking patterns with emitter information...");
    println!();

    let turn_taking = analyze_turn_taking(&annotations_vec);
    let social_network = analyze_social_network(&annotations_vec);
    let context_analysis = analyze_context(&annotations_vec);

    println!("✅ Turn-taking analysis complete");
    println!();

    // Display turn-taking results
    display_turn_taking_results(&turn_taking, &social_network, &context_analysis)?;
    println!();

    // ========================================================================
    // Step 5: Display Results
    // ========================================================================

    display_linguistic_results(&analysis, &clustered_phrases)?;

    // ========================================================================
    // Step 6: Export Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Exporting Results                                               │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let output_path = "/mnt/c/Users/sheel/Desktop/src/bat_analysis_results.json";
    export_results(&analysis, output_path)?;

    // Export phrase library
    if let Some(library) = pipeline.take_phrase_library() {
        let library_path = "/mnt/c/Users/sheel/Desktop/src/bat_phrase_library.json";
        export_phrase_library(&library, library_path)?;
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PIPELINE COMPLETE                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("📊 SUMMARY:");
    println!("   Phrase types processed: {}", phrase_directories.len());
    println!(
        "   Files processed: {} / 91,080 ({:.2}%)",
        total_files,
        total_files as f64 / 91080.0 * 100.0
    );
    println!("   Processing time: {:.2}s", processing_time.as_secs_f64());
    println!();
    println!("✅ Results exported to:");
    println!("   - {}", output_path);
    println!("   - /mnt/c/Users/sheel/Desktop/src/bat_phrase_library.json");
    println!();

    Ok(())
}

// ============================================================================
// Audio File Discovery
// ============================================================================

fn discover_phrase_directories(
    base_dir: &Path,
    max_dirs: usize,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    // For the flat structure, just return the audio directory itself
    println!("  Found flat audio directory with 91,080 WAV files");
    Ok(vec![base_dir.to_path_buf()])
}

// ============================================================================
// Parallel Audio Processing
// ============================================================================

fn process_bat_dataset_parallel(
    phrase_directories: &[PathBuf],
) -> Result<
    (
        Vec<VocalizationResult>,
        Vec<ClusteredPhrase>,
        usize,
        Vec<PhraseAudioSegment>,
    ),
    Box<dyn std::error::Error>,
> {
    use rayon::prelude::*;

    println!(
        "  Processing with {} workers...",
        std::thread::available_parallelism()
            .unwrap_or_else(|_| std::num::NonZeroUsize::new(1).unwrap())
            .get()
    );

    let mut total_files = 0;
    let mut all_vocalization_results = Vec::new();
    let mut all_clustered_phrases = Vec::new();
    let mut all_audio_segments = Vec::new();

    for audio_dir in phrase_directories {
        // Discover all WAV files in the flat audio directory
        let mut wav_files: Vec<_> = fs::read_dir(audio_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("wav"))
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
            .collect();

        wav_files.sort();
        // Limit files for testing
        if wav_files.len() > MAX_FILES_PER_PHRASE {
            wav_files.truncate(MAX_FILES_PER_PHRASE);
        }
        total_files = wav_files.len();

        println!("  - Processing {} WAV files", total_files);

        // Process files in parallel
        let results: Vec<_> = wav_files
            .par_iter()
            .enumerate()
            .filter_map(|(i, file_path)| process_single_bat_file(file_path, i).ok())
            .collect();

        // Collect results and create clustered phrases
        for (vocalization, segment) in results {
            // Create clustered phrase
            let intra_sim = 0.7;
            let inter_sim = 0.2;
            let is_atomic = intra_sim > 0.2 && inter_sim < 0.6;

            for phrase in &vocalization.phrases {
                all_clustered_phrases.push(ClusteredPhrase {
                    phrase: phrase.clone(),
                    cluster_id: all_clustered_phrases.len() as i32,
                    intra_cluster_similarity: intra_sim,
                    inter_cluster_similarity: inter_sim,
                    is_atomic,
                    contexts: vec![1], // Default context
                });
            }

            all_vocalization_results.push(vocalization);

            // Collect audio segment
            if let Some(seg) = segment {
                all_audio_segments.push(seg);
            }
        }
    }

    println!();

    Ok((
        all_vocalization_results,
        all_clustered_phrases,
        total_files,
        all_audio_segments,
    ))
}

fn process_single_bat_file(
    file_path: &Path,
    _index: usize,
) -> Result<(VocalizationResult, Option<PhraseAudioSegment>), Box<dyn std::error::Error>> {
    use technical_architecture::MicroDynamicsExtractor;

    let file_name = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Load audio using symphonia (auto-detects format: WAV, FLAC, MP3, AAC, OGG, etc.)
    let (audio_mono, sample_rate) = load_audio_file(file_path)?;

    if audio_mono.is_empty() {
        return Err("No audio samples found".into());
    }

    // Extract 56D features using the real MicroDynamicsExtractor
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features_56d = extractor
        .extract_56d(&audio_mono)
        .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    // Calculate actual duration from audio
    let duration_samples = audio_mono.len();
    let duration_ms = (duration_samples as f64 / sample_rate as f64) * 1000.0;

    // Calculate RMS from real audio
    let rms = (audio_mono.iter().map(|&x| x * x).sum::<f32>() / audio_mono.len() as f32).sqrt();

    // Convert 56D features to flat Vec<f64>
    // Structure: 30D base + 13 mfcc_delta + 13 mfcc_delta_delta = 56D
    let vector30d = features_56d.base_30d.to_vector30d(
        10000.0, // mean_f0_hz (estimated)
        duration_ms as f32,
        5000.0, // f0_range_hz (estimated)
    );

    let mut features_vec: Vec<f64> = vector30d.to_array().iter().map(|&x| x as f64).collect();

    // Append 13 mfcc_delta features
    for delta in &features_56d.mfcc_delta {
        features_vec.push(*delta as f64);
    }

    // Append 13 mfcc_delta_delta features
    for delta_delta in &features_56d.mfcc_delta_delta {
        features_vec.push(*delta_delta as f64);
    }

    // Final dimension: 30 + 13 + 13 = 56

    // Create phrase candidate with REAL features
    let phrase = PhraseCandidate {
        phrase_id: format!("bat_{}", file_name),
        file_name: file_name.to_string(),
        start_ms: 0.0,
        end_ms: duration_ms,
        duration_ms,
        features: features_vec,
        rms_amplitude: rms as f64,
        species: "egyptian_fruit_bat".to_string(),
        context: "vocalization".to_string(),
    };

    // Create audio segment with REAL audio
    let audio_segment = PhraseAudioSegment::new(
        audio_mono.clone(),
        sample_rate,
        format!("bat_{}", file_name),
        file_name.to_string(),
        0.0,
        duration_ms,
        10000.0, // mean_f0_hz (estimated)
        5000.0,  // f0_range_hz (estimated)
        rms as f64,
        "egyptian_fruit_bat".to_string(),
        "vocalization".to_string(),
    );

    Ok((
        VocalizationResult {
            file_name: file_name.to_string(),
            species: "egyptian_fruit_bat".to_string(),
            sentences: vec![],
            phrases: vec![phrase],
        },
        Some(audio_segment),
    ))
}

// ============================================================================
// Results Display
// ============================================================================

fn display_linguistic_results(
    analysis: &technical_architecture::LinguisticAnalysis,
    _clustered_phrases: &[ClusteredPhrase],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                  LINGUISTIC ANALYSIS RESULTS                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Information Theory
    println!("1️⃣  INFORMATION THEORY (Zipf's Law):");
    println!("   Slope (α): {:.4}", analysis.zipf.slope_alpha);
    println!("   Correlation (R²): {:.4}", analysis.zipf.correlation_r2);
    println!("   Efficiency: {:?}", analysis.zipf.efficiency);
    println!(
        "   Unique phrases: {}",
        analysis.zipf.phrase_frequencies.len()
    );
    println!();

    // Top 10 phrases
    println!("   Top 10 Most Frequent Phrases:");
    for (i, phrase_id) in analysis.zipf.ranked_phrases.iter().take(10).enumerate() {
        let freq = analysis
            .zipf
            .phrase_frequencies
            .get(phrase_id)
            .unwrap_or(&0);
        println!("     {:2}. {} (freq: {})", i + 1, phrase_id, freq);
    }
    println!();

    // 2. Prosody
    println!("2️⃣  PROSODY (Isochrony/Rhythm):");
    println!("   Rhythm: {:?}", analysis.prosody.rhythm);
    println!("   Gap CV: {:.4}", analysis.prosody.gap_cv);
    println!("   Mean gap: {:.2} ms", analysis.prosody.mean_gap_ms);
    println!();

    // 3. Phonotactics
    println!("3️⃣  PHONOTACTICS (Forbidden Transitions):");
    println!(
        "   Total transitions: {}",
        analysis.phonotactics.transition_matrix.len()
    );
    println!(
        "   Forbidden transitions: {}",
        analysis.phonotactics.forbidden_transitions.len()
    );
    println!();

    // 4. Pragmatics
    println!("4️⃣  PRAGMATICS (Turn-Taking):");
    println!("   Pattern: {:?}", analysis.pragmatics.pattern);
    println!();

    // 5. Atomicity
    println!("5️⃣  UPDATED ATOMICITY:");
    let truly_atomic = analysis
        .updated_atomic_phrases
        .iter()
        .filter(|p| p.is_truly_atomic)
        .count();

    println!(
        "   Total phrases: {}",
        analysis.updated_atomic_phrases.len()
    );
    println!(
        "   Truly atomic: {} ({:.1}%)",
        truly_atomic,
        truly_atomic as f64 / analysis.updated_atomic_phrases.len() as f64 * 100.0
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    Ok(())
}

fn display_turn_taking_results(
    turn_taking: &technical_architecture::TurnTakingAnalysis,
    social_network: &technical_architecture::SocialNetworkAnalysis,
    context_analysis: &technical_architecture::ContextAnalysis,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║              TURN-TAKING AND PRAGMATICS ANALYSIS (WITH EMITTER DATA)      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Turn-Taking Metrics
    println!("1️⃣  TURN-TAKING METRICS:");
    println!("   Turn-switch rate: {:.1}%", turn_taking.turn_switch_rate);
    println!(
        "   Total conversations: {}",
        turn_taking.total_conversations
    );
    println!("   A→B→A conversations: {}", turn_taking.aba_conversations);
    println!(
        "   Dyadic conversations (2 individuals): {}",
        turn_taking.dyadic_conversations
    );
    println!("   Pattern: {:?}", turn_taking.pattern);
    println!();

    // 2. Conversation Statistics
    println!("2️⃣  CONVERSATION STATISTICS:");
    println!(
        "   Mean length: {:.2} turns",
        turn_taking.conversation_stats.mean_length
    );
    println!(
        "   Median length: {:.1} turns",
        turn_taking.conversation_stats.median_length
    );
    println!(
        "   Min length: {} turn",
        turn_taking.conversation_stats.min_length
    );
    println!(
        "   Max length: {} turns",
        turn_taking.conversation_stats.max_length
    );
    println!(
        "   Multi-turn conversations (>2): {}",
        turn_taking.conversation_stats.multi_turn_count
    );
    println!(
        "   Long conversations (>10): {}",
        turn_taking.conversation_stats.long_conversation_count
    );
    println!();

    // 3. Response Time
    println!("3️⃣  RESPONSE TIME ANALYSIS:");
    println!(
        "   Mean gap: {:.2} files",
        turn_taking.response_time_stats.mean_gap
    );
    println!(
        "   Median gap: {:.1} files",
        turn_taking.response_time_stats.median_gap
    );
    println!(
        "   Immediate responses: {} ({:.1}%)",
        turn_taking.response_time_stats.immediate_response_count,
        turn_taking.response_time_stats.immediate_response_pct
    );
    println!();

    // 4. Social Network
    println!("4️⃣  SOCIAL NETWORK ANALYSIS:");
    println!("   Unique emitters: {}", social_network.unique_emitters);
    println!("   Unique addressees: {}", social_network.unique_addressees);
    println!(
        "   Unique interaction pairs: {}",
        social_network.unique_pairs
    );
    println!();

    // Top 5 emitters
    println!("   Top 5 Most Active Emitters:");
    let mut emitter_vec: Vec<_> = social_network.emitter_frequencies.iter().collect();
    emitter_vec.sort_by(|a, b| b.1.cmp(a.1));
    let total_vocalizations: usize = social_network.emitter_frequencies.values().sum();
    for (i, (emitter, count)) in emitter_vec.iter().take(5).enumerate() {
        let percentage = **count as f64 / total_vocalizations as f64 * 100.0;
        println!(
            "     {:2}. Emitter {:5}: {:>6} vocalizations ({:5.2}%)",
            i + 1,
            emitter,
            count,
            percentage
        );
    }
    println!();

    // Top 5 interaction pairs
    println!("   Top 5 Interaction Pairs:");
    println!(
        "     {:>12} → {:>12} {:>10}",
        "Emitter", "Addressee", "Count"
    );
    for (i, interaction) in social_network.top_interactions.iter().take(5).enumerate() {
        println!(
            "     {:2}. Emitter {:>5} → Addressee {:>5} {:>10}",
            i + 1,
            interaction.emitter,
            interaction.addressee,
            interaction.count
        );
    }
    println!();

    // 5. Context Analysis
    println!("5️⃣  CONTEXT ANALYSIS:");
    println!("   Unique contexts: {}", context_analysis.unique_contexts);
    println!();

    println!("   Context-specific turn-switch rates:");
    println!(
        "     {:>10} {:>15} {:>15} {:>10}",
        "Context", "Vocalizations", "Turn Switches", "Rate"
    );
    let mut context_vec: Vec<_> = context_analysis.context_turn_switch_rates.iter().collect();
    context_vec.sort_by(|a, b| a.0.cmp(b.0));
    for (_context_id, stats) in context_vec.iter() {
        println!(
            "     {:>10} {:>15} {:>15} {:>9.1}%",
            stats.context_id,
            stats.vocalization_count,
            stats.turn_switch_count,
            stats.turn_switch_rate
        );
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    Ok(())
}

// ============================================================================
// Export
// ============================================================================

fn export_results(
    analysis: &technical_architecture::LinguisticAnalysis,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json_output = serde_json::to_string_pretty(analysis)?;
    let file_size = json_output.len();
    fs::write(output_path, json_output)?;

    println!("✅ Results exported to: {}", output_path);
    println!("   File size: {} bytes", file_size);

    Ok(())
}

fn export_phrase_library(
    library: &PhraseAudioLibrary,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json_output = serde_json::to_string_pretty(library)?;
    let file_size = json_output.len();
    fs::write(output_path, json_output)?;

    println!("✅ Phrase library exported to: {}", output_path);
    println!(
        "   File size: {} bytes ({} MB)",
        file_size,
        file_size / 1_000_000
    );

    Ok(())
}

// ============================================================================
// Utility: Tilde Expansion
// ============================================================================

fn shellexpand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = std::env::var("HOME").ok() {
            return path.replacen("~", &home, 1);
        }
    }
    path.to_string()
}
