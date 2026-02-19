// Segment Marmoset Frame-Level Corpus into Proper Phrases
//
// This example converts the frame-level marmoset corpus into phrase-level
// by consolidating consecutive identical frames and detecting boundaries.
//
// Input: marmoset_corpus_for_analysis.json (frame-level labels)
// Output: marmoset_phrase_level_corpus.json (proper phrase sequences)
//
// Usage: cargo run --release --example segment_marmoset_corpus_to_phrases

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Segment Marmoset Frame-Level Corpus into Phrase-Level Sequences            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let input_path = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_corpus_for_analysis.json");
    let output_path = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_phrase_level_corpus.json");

    // ========================================================================
    // Step 1: Load Frame-Level Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Frame-Level Corpus                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let content = fs::read_to_string(input_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let sessions_array = json["sessions"].as_array().ok_or("Sessions not found")?;

    let cluster_to_phrase: HashMap<String, String> = json["cluster_to_phrase"]
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    println!("   📂 Loaded {} sessions", sessions_array.len());
    println!("   📊 Vocabulary size: {}", cluster_to_phrase.len());
    println!();

    // ========================================================================
    // Step 2: Segment Frames into Phrases
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Segmenting Frames into Phrases                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   🔄 Processing each session...");

    let mut phrase_level_sessions: Vec<PhraseLevelSession> = Vec::new();
    let mut total_phrases = 0usize;
    let mut total_frames = 0usize;

    for (session_idx, session_data) in sessions_array.iter().enumerate() {
        let frames: Vec<i32> = if let Some(arr) = session_data.as_array() {
            arr.iter()
                .filter_map(|v| v.as_i64())
                .map(|v| v as i32)
                .collect()
        } else {
            continue;
        };

        total_frames += frames.len();

        // Convert frames to phrase-level by consolidating consecutive runs
        let phrases = consolidate_frames_to_phrases(&frames);

        total_phrases += phrases.len();

        phrase_level_sessions.push(PhraseLevelSession {
            session_id: session_idx,
            call_type: "Vocalization".to_string(), // Default context
            phrases,
            original_frame_count: frames.len(),
        });

        if (session_idx + 1) % 5 == 0 {
            println!(
                "      Processed {}/{} sessions",
                session_idx + 1,
                sessions_array.len()
            );
        }
    }

    println!();
    println!("   ✅ Segmentation complete");
    println!();

    // ========================================================================
    // Step 3: Statistics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Segmentation Statistics                                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let avg_phrases_per_session = total_phrases as f64 / phrase_level_sessions.len() as f64;
    let avg_frames_per_phrase = total_frames as f64 / total_phrases as f64;

    println!("   📊 Statistics:");
    println!("      ├─ Total sessions: {}", phrase_level_sessions.len());
    println!(
        "      ├─ Total phrases (after segmentation): {}",
        total_phrases
    );
    println!(
        "      ├─ Average phrases per session: {:.1}",
        avg_phrases_per_session
    );
    println!(
        "      ├─ Average frames per phrase: {:.1}",
        avg_frames_per_phrase
    );
    println!();

    // Count phrase frequencies across all sessions
    let mut phrase_freq: HashMap<i32, usize> = HashMap::new();
    for session in &phrase_level_sessions {
        for &phrase_id in &session.phrases {
            *phrase_freq.entry(phrase_id).or_insert(0) += 1;
        }
    }

    let unique_phrases = phrase_freq.len();
    println!("      ├─ Unique phrase types: {}", unique_phrases);
    println!();

    // Most common phrases
    let mut freq_vec: Vec<_> = phrase_freq.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("   📚 Top 20 Most Common Phrases:");
    for (i, (phrase_id, count)) in freq_vec.iter().take(20).enumerate() {
        let desc = cluster_to_phrase
            .get(&phrase_id.to_string())
            .map(|s| s.as_str())
            .unwrap_or("Unknown");

        println!(
            "      {:2}. Phrase {:4} ({}): {} occurrences",
            i + 1,
            phrase_id,
            truncate(desc, 30),
            count
        );
    }
    println!();

    // ========================================================================
    // Step 4: Save Phrase-Level Corpus
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Saving Phrase-Level Corpus                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let output_corpus = PhraseLevelCorpus {
        metadata: CorpusMetadata {
            description: "Phrase-level marmoset corpus (segmented from frame-level)".to_string(),
            num_sessions: phrase_level_sessions.len(),
            total_phrases,
            vocabulary_size: unique_phrases,
            species: "marmoset".to_string(),
            segmentation_method: "consecutive_frame_consolidation".to_string(),
        },
        sessions: phrase_level_sessions,
        cluster_to_phrase,
    };

    let output_json = serde_json::to_string_pretty(&output_corpus)?;
    fs::write(output_path, output_json)?;

    println!("   💾 Saved phrase-level corpus to:");
    println!("      {}", output_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SEGMENTATION COMPLETE                                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 RESULTS:                                                             ║");
    println!("║     • Frame-level corpus converted to phrase-level                         ║");
    println!(
        "║     • Sessions: {}                                                        ║",
        phrase_level_sessions.len()
    );
    println!(
        "║     • Total phrases: {}                                                   ║",
        total_phrases
    );
    println!(
        "║     • Unique vocabulary: {}                                               ║",
        unique_phrases
    );
    println!("║                                                                           ║");
    println!("║  📁 Output file:                                                           ║");
    println!(
        "║     {}                                              ║",
        output_path.display()
    );
    println!("║                                                                           ║");
    println!("║  ✅ This phrase-level corpus can now be used with:                         ║");
    println!("║     • phrase_context_analysis_marmoset_generality.rs                     ║");
    println!("║     • phase2_advanced_sequence_analysis_marmoset.rs                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

/// Consolidate consecutive identical frames into phrase units
fn consolidate_frames_to_phrases(frames: &[i32]) -> Vec<i32> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut phrases = Vec::new();
    let mut current_phrase = frames[0];
    let mut run_length = 1usize;
    let min_phrase_frames = 3; // Minimum frames to consider a valid phrase

    for &frame in &frames[1..] {
        if frame == current_phrase {
            run_length += 1;
        } else {
            // End of current run - add as phrase if long enough
            if run_length >= min_phrase_frames {
                phrases.push(current_phrase);
            }
            // Start new run
            current_phrase = frame;
            run_length = 1;
        }
    }

    // Don't forget the last run
    if run_length >= min_phrase_frames {
        phrases.push(current_phrase);
    }

    phrases
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseLevelCorpus {
    metadata: CorpusMetadata,
    sessions: Vec<PhraseLevelSession>,
    cluster_to_phrase: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusMetadata {
    description: String,
    num_sessions: usize,
    total_phrases: usize,
    vocabulary_size: usize,
    species: String,
    segmentation_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseLevelSession {
    session_id: usize,
    call_type: String,
    phrases: Vec<i32>, // Phrase-level sequence (not frame-level)
    original_frame_count: usize,
}
