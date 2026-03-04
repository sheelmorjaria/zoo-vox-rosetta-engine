//! Upgrade NBD Cache to Level 3 (Interaction-Ready)
//! =================================================
//!
//! Fast approach: Join existing NBD cache with original annotations CSV
//! to add Addressee (Receiver ID) without re-extracting audio features.
//!
//! This upgrades from Level 2.5 → Level 3 in seconds instead of hours.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Original annotation from CSV
#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    emitter: i32,
    addressee: i32,
    context: i32,
}

/// Upgraded segment with interaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpgradedSegment {
    source_file: String,
    context: i32,
    emitter: i32,
    addressee: i32, // NEW: Receiver ID
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    boundary_type: String,
    features: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     CACHE UPGRADE: Level 2.5 → Level 3 (Interaction-Ready)               ║");
    println!("║     Adding Addressee ID via CSV Join                                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_nbd_cache_normalized");
    let output_dir = Path::new("bat_nbd_cache_level3");
    let annotations_path =
        Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv");

    fs::create_dir_all(output_dir)?;

    // ---------------------------------------------------------
    // STEP 1: LOAD ANNOTATIONS
    // ---------------------------------------------------------
    println!("[1/3] Loading Original Annotations...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let csv_content = fs::read_to_string(annotations_path)?;
    let mut annotations: HashMap<String, Annotation> = HashMap::new();

    for (i, line) in csv_content.lines().enumerate() {
        if i == 0 {
            continue;
        } // Skip header

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let emitter: i32 = parts[0].parse().unwrap_or(0);
            let addressee: i32 = parts[1].parse().unwrap_or(0);
            let context: i32 = parts[2].parse().unwrap_or(0);
            let filename = parts[7].to_string();

            annotations.insert(
                filename,
                Annotation {
                    emitter,
                    addressee,
                    context,
                },
            );
        }
    }

    println!("  Loaded {} file annotations", annotations.len());
    println!();

    // ---------------------------------------------------------
    // STEP 2: PROCESS CACHE FILES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/3] Upgrading Cache Files...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("  Found {} cache files", cache_files.len());

    // Statistics
    let mut total_segments = 0usize;
    let mut matched_segments = 0usize;
    let mut emitter_counts: HashMap<i32, usize> = HashMap::new();
    let mut addressee_counts: HashMap<i32, usize> = HashMap::new();
    let mut interaction_pairs: HashMap<(i32, i32), usize> = HashMap::new();
    let mut self_addressed = 0usize;

    // Process in parallel
    let results: Vec<_> = cache_files
        .par_iter()
        .map(|cache_file| {
            let json = fs::read_to_string(cache_file).ok()?;
            let batch: Vec<serde_json::Value> = serde_json::from_str(&json).ok()?;

            let mut upgraded: Vec<UpgradedSegment> = Vec::new();
            let mut local_matched = 0usize;
            let mut local_emitters: HashMap<i32, usize> = HashMap::new();
            let mut local_addressees: HashMap<i32, usize> = HashMap::new();
            let mut local_pairs: HashMap<(i32, i32), usize> = HashMap::new();
            let mut local_self = 0usize;

            for seg in batch {
                let source_file = seg.get("source_file")?.as_str()?.to_string();
                let annotation = annotations.get(&source_file);

                let (emitter, addressee, context) = match annotation {
                    Some(a) => {
                        local_matched += 1;
                        (a.emitter, a.addressee, a.context)
                    }
                    None => {
                        let ctx = seg.get("context")?.as_i64()? as i32;
                        let emit = seg.get("emitter")?.as_i64()? as i32;
                        (emit, -1, ctx)
                    }
                };

                let segment = UpgradedSegment {
                    source_file: source_file.clone(),
                    context,
                    emitter,
                    addressee,
                    segment_idx: seg.get("segment_idx")?.as_u64()? as usize,
                    start_ms: seg.get("start_ms")?.as_f64()? as f32,
                    end_ms: seg.get("end_ms")?.as_f64()? as f32,
                    boundary_type: seg.get("boundary_type")?.as_str()?.to_string(),
                    features: seg
                        .get("features")?
                        .as_array()?
                        .iter()
                        .filter_map(|v| v.as_f64().map(|x| x as f32))
                        .collect(),
                };

                // Stats
                *local_emitters.entry(emitter).or_insert(0) += 1;
                *local_addressees.entry(addressee).or_insert(0) += 1;
                if emitter != addressee && addressee != -1 {
                    *local_pairs.entry((emitter, addressee)).or_insert(0) += 1;
                } else if emitter == addressee {
                    local_self += 1;
                }

                upgraded.push(segment);
            }

            Some((
                cache_file.clone(),
                upgraded,
                local_matched,
                local_emitters,
                local_addressees,
                local_pairs,
                local_self,
            ))
        })
        .filter_map(|x| x)
        .collect();

    // Aggregate results
    for (
        cache_file,
        upgraded,
        local_matched,
        local_emitters,
        local_addressees,
        local_pairs,
        local_self,
    ) in results
    {
        total_segments += upgraded.len();
        matched_segments += local_matched;

        // Merge stats
        for (k, v) in local_emitters {
            *emitter_counts.entry(k).or_insert(0) += v;
        }
        for (k, v) in local_addressees {
            *addressee_counts.entry(k).or_insert(0) += v;
        }
        for (k, v) in local_pairs {
            *interaction_pairs.entry(k).or_insert(0) += v;
        }
        self_addressed += local_self;

        // Save upgraded batch
        let filename = cache_file.file_name().unwrap().to_str().unwrap();
        let output_file = output_dir.join(filename);
        let json = serde_json::to_string(&upgraded)?;
        fs::write(&output_file, json)?;

        if total_segments % 100000 == 0 {
            println!("  Processed {} segments...", total_segments);
        }
    }

    println!(
        "  Upgraded {} segments from {} files",
        total_segments,
        cache_files.len()
    );
    println!(
        "  Matched with annotations: {} ({:.1}%)",
        matched_segments,
        matched_segments as f64 / total_segments as f64 * 100.0
    );
    println!();

    // ---------------------------------------------------------
    // STEP 3: SUMMARY
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/3] Level 3 Upgrade Summary");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Interaction statistics
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  INTERACTION STATISTICS                                                 │");

    // Top emitters
    let mut sorted_emitters: Vec<_> = emitter_counts.iter().collect();
    sorted_emitters.sort_by(|a, b| b.1.cmp(a.1));
    println!("  │  Top Emitters:                                                          │");
    for (emit, count) in sorted_emitters.iter().take(5) {
        let pct = **count as f64 / total_segments as f64 * 100.0;
        println!(
            "  │    • Bat {:5}: {:7} calls ({:.1}%)                              │",
            emit, count, pct
        );
    }

    // Top addressees
    let mut sorted_addressees: Vec<_> = addressee_counts.iter().collect();
    sorted_addressees.sort_by(|a, b| b.1.cmp(a.1));
    println!("  │  Top Addressees:                                                        │");
    for (addr, count) in sorted_addressees.iter().take(5) {
        let pct = **count as f64 / total_segments as f64 * 100.0;
        println!(
            "  │    • Bat {:5}: {:7} received ({:.1}%)                            │",
            addr, count, pct
        );
    }

    println!("  │                                                                          │");
    println!(
        "  │  Self-addressed: {} ({:.1}%)                                             │",
        self_addressed,
        self_addressed as f64 / total_segments as f64 * 100.0
    );

    // Top interaction pairs
    let mut sorted_pairs: Vec<_> = interaction_pairs.iter().collect();
    sorted_pairs.sort_by(|a, b| b.1.cmp(a.1));
    println!("  │  Top Interaction Pairs (Emitter → Addressee):                           │");
    for ((emit, addr), count) in sorted_pairs.iter().take(5) {
        let pct = **count as f64 / total_segments as f64 * 100.0;
        println!(
            "  │    • {:5} → {:5}: {:7} calls ({:.1}%)                          │",
            emit, addr, count, pct
        );
    }

    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Level 3 capability
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  LEVEL 3 CAPABILITY UNLOCKED                                            │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │                                                                          │");
    println!(
        "  │  ✓ Total Segments: {:>8}                                             ",
        total_segments
    );
    println!(
        "  │  ✓ Unique Emitters: {:>8}                                             ",
        emitter_counts.len()
    );
    println!(
        "  │  ✓ Unique Addressees: {:>8}                                            ",
        addressee_counts.len()
    );
    println!(
        "  │  ✓ Interaction Pairs: {:>8}                                            ",
        interaction_pairs.len()
    );
    println!("  │                                                                          │");
    println!("  │  ═══════════════════════════════════════════════════════════════════    │");
    println!("  │                                                                          │");
    println!("  │  NOW POSSIBLE:                                                           │");
    println!("  │  • Turn-Taking Models: \"If Bat A calls, who replies?\"                  │");
    println!("  │  • Addressing Analysis: \"Who talks to whom?\"                           │");
    println!("  │  • Social Network: Build interaction graphs                            │");
    println!("  │  • Response Timing: \"How fast do they reply?\"                         │");
    println!("  │                                                                          │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Output: {}", output_dir.display());
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
