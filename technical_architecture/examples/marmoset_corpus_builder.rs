// Marmoset Corpus Builder - Creates Phrase-Level Corpus from FLAC Files
//
// This example scans the marmoset Vocalizations directory and creates a
// phrase-level corpus with call types as contexts.
//
// Usage: cargo run --release --example marmoset_corpus_builder

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═════════════════════════════════════════════════════════════════╗");
    println!("║    Marmoset Corpus Builder                                            ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                       ║");
    println!("║  Scanning Vocalizations directory and building phrase-level corpus     ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();

    let start_time = Instant::now();

    // Configuration
    let vocalizations_dir = PathBuf::from("/home/sheel/birdsong_analysis/data/Vocalizations");
    let output_path =
        PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phrase_level_corpus.json");
    let output_symbolic_stream =
        PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_symbolic_stream.json");

    // ========================================================================
    // Step 1: Scan Vocalizations Directory
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Scanning Vocalizations Directory                             │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    if !vocalizations_dir.exists() {
        println!(
            "   ❌ Vocalizations directory not found: {}",
            vocalizations_dir.display()
        );
        return Err("Directory not found".into());
    }

    // Scan all date subdirectories and collect FLA Cs
    let mut call_type_files: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let mut total_files = 0usize;

    let entries = fs::read_dir(&vocalizations_dir)?;
    for entry in entries {
        let entry = entry?;
        let entry_path = entry.path();

        // Each date directory contains FLAC files
        if entry_path.is_dir() {
            if let Ok(files_iter) = fs::read_dir(&entry_path) {
                let files: Vec<_> = files_iter.filter_map(|f| f.ok()).collect();
                for file_entry in files {
                    let file_path = file_entry.path();
                    if file_path.extension().and_then(|s| s.to_str()) == Some("flac") {
                        if let Some(file_name_os) = file_path.file_name() {
                            if let Some(file_name) = file_name_os.to_str() {
                                // Extract call type from filename (e.g., "Vocalization_12345.flac")
                                let call_type = extract_call_type(file_name);

                                call_type_files
                                    .entry(call_type.clone())
                                    .or_insert_with(Vec::new)
                                    .push(file_path.clone());

                                total_files += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    println!("   📂 Found {} FLAC files", total_files);
    println!();
    println!("   Call Types:");
    let mut call_types: Vec<_> = call_type_files.iter().collect();
    call_types.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (call_type, files) in &call_types {
        println!("      • {}: {} files", call_type, files.len());
    }
    println!();

    // ========================================================================
    // Step 2: Create Symbolic Stream (phrase IDs by context)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Building Symbolic Stream                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Assign unique phrase IDs to each file based on call type
    let mut phrase_id_counter = 0i32;
    let mut file_to_phrase: HashMap<String, i32> = HashMap::new();
    let mut cluster_to_phrase: HashMap<String, String> = HashMap::new();
    let mut sessions: Vec<serde_json::Value> = Vec::new();

    // Group files by call type and create sessions
    for (call_type, files) in &call_types {
        let phrase_id_start = phrase_id_counter;
        let mut phrases_in_session = Vec::new();

        for file_path in files.iter() {
            if let Some(file_name_os) = file_path.file_name() {
                if let Some(file_name) = file_name_os.to_str() {
                    let phrase_id = phrase_id_counter;
                    phrase_id_counter += 1;

                    file_to_phrase.insert(file_name.to_string(), phrase_id);
                    phrases_in_session.push(phrase_id);

                    // Map cluster (filename without extension) to phrase ID
                    let cluster_name = file_name.replace(".flac", "");
                    cluster_to_phrase.insert(cluster_name, format!("{}_{}", call_type, phrase_id));
                }
            }
        }

        // Create a session for this call type with all its phrases
        sessions.push(serde_json::json!({
            "call_type": call_type,
            "phrases": phrases_in_session,
        }));

        println!(
            "   📝 {} session: phrases {}-{} ({})",
            call_type,
            phrase_id_start,
            phrase_id_counter - 1,
            phrases_in_session.len()
        );
    }

    println!();

    // ========================================================================
    // Step 3: Build Phrase-Level Corpus Output
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Creating Phrase-Level Corpus                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let total_phrases = sessions
        .iter()
        .map(|s| s["phrases"].as_array().map(|a| a.len()).unwrap_or(0))
        .sum::<usize>();

    let corpus = serde_json::json!({
        "metadata": {
            "description": "Phrase-level marmoset corpus from FLAC files",
            "species": "marmoset",
            "source_dir": vocalizations_dir.display().to_string(),
            "num_sessions": sessions.len(),
            "total_phrases": total_phrases,
            "vocabulary_size": phrase_id_counter,
            "call_types": call_types.iter().map(|(k, v)| (k, v.len())).collect::<HashMap<_, _>>()
        },
        "sessions": sessions,
        "cluster_to_phrase": cluster_to_phrase,
    });

    // ========================================================================
    // Step 4: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Saving Results                                               │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    fs::write(&output_path, serde_json::to_string_pretty(&corpus)?)?;
    println!("   💾 Phrase-level corpus: {}", output_path.display());

    // Also save symbolic stream format for sequence analysis
    let symbolic_stream = serde_json::json!({
        "metadata": corpus["metadata"],
        "symbolic_streams": sessions,
    });

    fs::write(
        &output_symbolic_stream,
        serde_json::to_string_pretty(&symbolic_stream)?,
    )?;
    println!(
        "   💾 Symbolic stream: {}",
        output_symbolic_stream.display()
    );
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    CORPUS BUILD COMPLETE                               ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║                                                                       ║");
    println!("║  📊 SUMMARY:                                                         ║");
    println!(
        "║     • Total files scanned: {}                                       ║",
        total_files
    );
    println!(
        "║     • Call types: {}                                                ║",
        call_types.len()
    );
    println!(
        "║     • Total phrases: {}                                              ║",
        phrase_id_counter
    );
    println!(
        "║     • Build time: {:.2}s                                             ║",
        elapsed.as_secs_f64()
    );
    println!("║                                                                       ║");
    println!("║  📁 Output files:                                                    ║");
    println!(
        "║     • {}                                               ║",
        output_path.display()
    );
    println!(
        "║     • {}                                        ║",
        output_symbolic_stream.display()
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

/// Extract call type from filename (e.g., "Vocalization_12345.flac" -> "Vocalization")
fn extract_call_type(filename: &str) -> String {
    // Call types: Vocalization, Trill, Seep, Tsik, Twitter, Phee, Infant_cry
    if filename.starts_with("Vocalization_") {
        "Vocalization".to_string()
    } else if filename.starts_with("Trill_") {
        "Trill".to_string()
    } else if filename.starts_with("Seep_") {
        "Seep".to_string()
    } else if filename.starts_with("Tsik_") {
        "Tsik".to_string()
    } else if filename.starts_with("Twitter_") {
        "Twitter".to_string()
    } else if filename.starts_with("Phee_") {
        "Phee".to_string()
    } else if filename.starts_with("Infant_cry_") {
        "Infant_cry".to_string()
    } else {
        "Unknown".to_string()
    }
}
