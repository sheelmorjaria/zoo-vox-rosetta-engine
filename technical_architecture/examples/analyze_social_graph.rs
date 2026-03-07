//! Bat Social Graph & Turn-Taking Analysis
//! =========================================
//!
//! Level 3 enables:
//! - Social Network Analysis: Who talks to whom
//! - Dyad Dialects: Unique patterns between specific pairs
//! - Turn-Taking Prediction: Response timing and patterns
//! - Targeted Playback: "Digital Ventriloquism"

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct Segment {
    source_file: String,
    context: i32,
    emitter: i32,
    addressee: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    #[allow(dead_code)]
    boundary_type: String,
    features: Vec<f32>,
}

/// Social interaction statistics
struct InteractionStats {
    total_calls: usize,
    unique_emitters: usize,
    unique_addressees: usize,
    interaction_pairs: usize,
    hubs: Vec<(i32, usize, usize)>,  // (bat_id, sent, received)
    dyads: Vec<((i32, i32), usize)>, // ((emitter, addressee), count)
    self_talk: usize,
}

/// Turn-taking pattern
#[derive(Debug, Clone)]
struct TurnPattern {
    emitter: i32,
    addressee: i32,
    sequence: Vec<Vec<f32>>, // Consecutive feature vectors
    contexts: Vec<i32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     BAT SOCIAL GRAPH & TURN-TAKING ANALYSIS                              ║");
    println!("║     Level 3: Who talks to Whom, and Why?                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_nbd_cache_level3");

    // ---------------------------------------------------------
    // STEP 1: LOAD ALL SEGMENTS
    // ---------------------------------------------------------
    println!("[1/3] Loading Level 3 Cache...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let all_segments: Vec<Segment> = cache_files
        .par_iter()
        .flat_map(|file| {
            let json = fs::read_to_string(file).ok()?;
            let batch: Vec<Segment> = serde_json::from_str(&json).ok()?;
            Some(batch)
        })
        .flatten()
        .collect();

    println!("  Loaded {} segments", all_segments.len());
    println!();

    // ---------------------------------------------------------
    // STEP 2: BUILD SOCIAL GRAPH
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/3] Social Graph Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Count calls by emitter and addressee
    let mut sent: HashMap<i32, usize> = HashMap::new();
    let mut received: HashMap<i32, usize> = HashMap::new();
    let mut pairs: HashMap<(i32, i32), usize> = HashMap::new();
    let mut self_talk = 0usize;

    for seg in &all_segments {
        *sent.entry(seg.emitter).or_insert(0) += 1;
        *received.entry(seg.addressee).or_insert(0) += 1;

        if seg.emitter != seg.addressee {
            *pairs.entry((seg.emitter, seg.addressee)).or_insert(0) += 1;
        } else {
            self_talk += 1;
        }
    }

    // Find hubs (high received, may be moderate sent)
    let mut hubs: Vec<_> = sent
        .keys()
        .map(|&id| {
            let s = sent.get(&id).copied().unwrap_or(0);
            let r = received.get(&id).copied().unwrap_or(0);
            (id, s, r)
        })
        .collect();
    hubs.sort_by(|a, b| (b.2).cmp(&a.2)); // Sort by received

    // Find dyads (strongest pairs)
    let mut dyads: Vec<_> = pairs.iter().collect();
    dyads.sort_by(|a, b| b.1.cmp(a.1));

    // Print analysis
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  COLONY OVERVIEW                                                        │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Total vocalizations: {:>8}                                        ",
        all_segments.len()
    );
    println!(
        "  │  Unique emitters:     {:>8}                                        ",
        sent.len()
    );
    println!(
        "  │  Unique addressees:   {:>8}                                        ",
        received.len()
    );
    println!(
        "  │  Interaction pairs:   {:>8}                                        ",
        pairs.len()
    );
    println!(
        "  │  Self-talk:           {:>8} ({:.1}%)                               ",
        self_talk,
        self_talk as f64 / all_segments.len() as f64 * 100.0
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Hubs analysis
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  SOCIAL HUBS (Most Addressed)                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for (id, sent_count, recv_count) in hubs.iter().take(10) {
        let recv_pct = *recv_count as f64 / all_segments.len() as f64 * 100.0;
        let sent_pct = *sent_count as f64 / all_segments.len() as f64 * 100.0;
        let role = if *recv_count > 100000 {
            "🎯 CENTRAL"
        } else if *recv_count > 50000 {
            "★ Major"
        } else {
            ""
        };
        println!(
            "  │  Bat {:6}: Sent {:6} ({:4.1}%) | Recv {:7} ({:4.1}%) {}",
            id, sent_count, sent_pct, recv_count, recv_pct, role
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Dyads analysis
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  TOP DYADS (Bonded Pairs / Rivals)                                     │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for ((emit, addr), count) in dyads.iter().take(15) {
        let pct = **count as f64 / all_segments.len() as f64 * 100.0;
        let bidirectional = pairs.get(&(*addr, *emit)).copied().unwrap_or(0);

        let relationship = if bidirectional > 20000 {
            "💕 Strong Bond"
        } else if bidirectional > 10000 {
            "↔ Frequent Exchange"
        } else if **count > 30000 {
            "📢 One-way Broadcast"
        } else {
            ""
        };

        println!(
            "  │  {:6} → {:6}: {:7} calls ({:4.1}%) [↔{:6}] {}",
            emit, addr, count, pct, bidirectional, relationship
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ---------------------------------------------------------
    // STEP 3: TURN-TAKING OPPORTUNITIES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/3] Turn-Taking Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Group segments by file to find sequences
    let mut by_file: HashMap<String, Vec<&Segment>> = HashMap::new();
    for seg in &all_segments {
        by_file.entry(seg.source_file.clone()).or_default().push(seg);
    }

    // Find turn-taking patterns (A→B transitions)
    let mut turn_patterns: HashMap<(i32, i32), usize> = HashMap::new();
    let mut response_times: Vec<f32> = Vec::new();

    for (_, segs) in by_file.iter() {
        let mut sorted = segs.clone();
        sorted.sort_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap());

        for i in 0..sorted.len().saturating_sub(1) {
            let curr = sorted[i];
            let next = sorted[i + 1];

            // Check if this is a "turn" (different emitters)
            if curr.emitter != next.emitter {
                // curr.addressee == next.emitter means "A called B, B responded"
                if curr.addressee == next.emitter {
                    *turn_patterns.entry((curr.emitter, next.emitter)).or_insert(0) += 1;
                    response_times.push(next.start_ms - curr.end_ms);
                }
            }
        }
    }

    let mut turns: Vec<_> = turn_patterns.iter().collect();
    turns.sort_by(|a, b| b.1.cmp(a.1));

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  TURN-TAKING PATTERNS (A→B Response Chains)                            │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for ((from, to), count) in turns.iter().take(10) {
        println!("  │  {:6} → {:6}: {:6} response chains", from, to, count);
    }

    let avg_response = if !response_times.is_empty() {
        response_times.iter().sum::<f32>() / response_times.len() as f32
    } else {
        0.0
    };

    let median_response = {
        let mut sorted = response_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        }
    };

    println!("  │                                                                          │");
    println!("  │  Response Timing:                                                        │");
    println!(
        "  │    • Average: {:.1}ms                                               ",
        avg_response
    );
    println!(
        "  │    • Median:  {:.1}ms                                               ",
        median_response
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Targeted playback suggestions
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  TARGETED PLAYBACK SUGGESTIONS                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    // Find the best pair for targeted playback
    if let Some(((best_from, best_to), count)) = turns.first() {
        println!("  │                                                                          │");
        println!("  │  🎯 Best pair for playback testing:                                      │");
        println!(
            "  │     Emitter {:6} → Target {:6} ({} response chains)            ",
            best_from, best_to, count
        );
        println!("  │                                                                          │");
        println!("  │  Strategy:                                                               │");
        println!(
            "  │    1. Synthesize calls using patterns from Emitter {:6}            ",
            best_from
        );
        println!(
            "  │    2. Target {:6} is most likely to respond                      ",
            best_to
        );
        println!(
            "  │    3. Expected response time: ~{:.0}ms                                  ",
            median_response
        );
    }

    println!("  │                                                                          │");
    println!("  │  💡 Social Engineering (Digital Ventriloquism):                         │");
    println!("  │                                                                          │");

    // Find a strong dyad
    if let Some(((emit, addr), count)) = dyads.first() {
        println!(
            "  │    Pair {:6} ↔ {:6} has {} interactions                 ",
            emit, addr, count
        );
        println!(
            "  │    → Impersonate {:6} to attract {:6}                      ",
            emit, addr
        );
    }

    println!("  │                                                                          │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
