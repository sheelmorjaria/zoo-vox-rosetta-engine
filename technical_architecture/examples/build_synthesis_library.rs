//! Build Rosetta Synthesis Library
//! ================================
//!
//! Creates a "Clean Atomic Library" from NBD segments:
//! 1. ATOMIC LAYER (Grains): Representative sounds for each acoustic state
//! 2. SYNTACTIC LAYER (Templates): High-purity N-gram patterns
//! 3. METADATA: Feature vectors, state IDs, context probabilities
//!
//! This creates a "Lego Set" for animal communication:
//! - Finite blocks (Grains) → Infinite sentences (Templates)

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

type StateId = u32;

#[derive(Debug, Clone, Deserialize)]
struct CachedSeg {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    #[allow(dead_code)]
    boundary_type: String,
    features: Vec<f32>,
}

/// A single "grain" - representative audio for an acoustic state
#[derive(Serialize, Deserialize)]
struct GrainEntry {
    id: usize,
    state_id: StateId,
    source_file: String,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    centroid_features: Vec<f32>,
    dominant_context: i32,
    context_purity: f64,
    context_distribution: HashMap<i32, f64>,
    sample_count: usize,
}

/// A syntactic template - reusable sequence pattern
#[derive(Serialize, Deserialize)]
struct SyntaxTemplate {
    id: usize,
    pattern: Vec<StateId>,
    pattern_str: String,
    n: usize,
    dominant_context: i32,
    purity: f64,
    total_occurrences: usize,
    context_distribution: HashMap<i32, f64>,
    grain_ids: Vec<usize>,
}

/// N-gram statistics for context mapping
struct NgramStats {
    total_count: usize,
    context_counts: HashMap<i32, usize>,
}

/// Library manifest
#[derive(Serialize)]
struct LibraryManifest {
    total_segments: usize,
    total_files: usize,
    n_acoustic_states: usize,
    n_grains: usize,
    n_templates: usize,
    discretization_bins: usize,
    created: String,
    grains: Vec<GrainEntry>,
    templates: Vec<SyntaxTemplate>,
    context_vocabulary: HashMap<i32, ContextVocabulary>,
}

#[derive(Serialize)]
struct ContextVocabulary {
    context_id: i32,
    n_patterns: usize,
    top_patterns: Vec<String>,
    total_occurrences: usize,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     ROSETTA SYNTHESIS LIBRARY BUILDER                                     ║");
    println!("║     Creating 'Grains' and 'Templates' for Two-Way Communication           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_nbd_cache_normalized");
    let output_dir = Path::new("rosetta_synthesis_library");

    if !cache_dir.exists() {
        eprintln!("Error: Cache not found: {}", cache_dir.display());
        std::process::exit(1);
    }

    fs::create_dir_all(output_dir)?;

    // ---------------------------------------------------------
    // PHASE 1: LOAD & STATE ASSIGNMENT
    // ---------------------------------------------------------
    println!("[1/5] Loading Segments and Assigning Acoustic States...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let all_segments: Vec<CachedSeg> = cache_files
        .par_iter()
        .flat_map(|file| {
            let json = match fs::read_to_string(file) {
                Ok(j) => j,
                Err(_) => return None,
            };
            let batch: Vec<CachedSeg> = match serde_json::from_str(&json) {
                Ok(b) => b,
                Err(_) => return None,
            };
            Some(batch)
        })
        .flatten()
        .collect();

    let total_segments = all_segments.len();
    let unique_files: std::collections::HashSet<_> = all_segments.iter().map(|s| s.source_file.clone()).collect();

    println!("  Loaded {} segments from {} files", total_segments, unique_files.len());

    // Discretize into acoustic states
    let pitch_indices: Vec<usize> = vec![0, 1, 2, 40, 41, 42];
    let n_features = pitch_indices.len();
    let bins_per_dim = 3;

    let mut features: Vec<Vec<f64>> = all_segments
        .iter()
        .map(|seg| {
            pitch_indices
                .iter()
                .map(|&idx| {
                    if idx < seg.features.len() {
                        seg.features[idx] as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    // Z-normalize
    for j in 0..n_features {
        let col: Vec<f64> = features.iter().map(|f| f[j]).collect();
        let mean = col.iter().sum::<f64>() / total_segments as f64;
        let std = (col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / total_segments as f64)
            .sqrt()
            .max(1e-8);
        for f in features.iter_mut() {
            f[j] = (f[j] - mean) / std;
        }
    }

    // Grid discretization
    let state_ids: Vec<StateId> = features
        .iter()
        .map(|f| {
            let mut state = 0u32;
            let mut multiplier = 1u32;
            for &val in f {
                let bin = if val < -0.5 {
                    0
                } else if val < 0.5 {
                    1
                } else {
                    2
                };
                state += bin * multiplier;
                multiplier *= bins_per_dim as u32;
            }
            state
        })
        .collect();

    // Group segments by state
    let mut state_buckets: HashMap<StateId, Vec<usize>> = HashMap::new();
    for (idx, &state) in state_ids.iter().enumerate() {
        state_buckets.entry(state).or_default().push(idx);
    }

    let n_states = state_buckets.len();
    println!("  Discretized into {} acoustic states", n_states);
    println!();

    // ---------------------------------------------------------
    // PHASE 2: BUILD ATOMIC LIBRARY (Grains)
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/5] Extracting Representative Grains...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let mut grain_library: Vec<GrainEntry> = Vec::new();

    for (state_id, indices) in &state_buckets {
        // 1. Calculate centroid features
        let mut sum_features = vec![0.0f64; n_features];
        for &idx in indices {
            for (j, &val) in features[idx].iter().enumerate() {
                sum_features[j] += val;
            }
        }
        let centroid: Vec<f32> = sum_features
            .iter()
            .map(|&x| (x / indices.len() as f64) as f32)
            .collect();

        // 2. Find segment closest to centroid
        let mut best_idx = indices[0];
        let mut best_dist = f64::MAX;

        for &idx in indices.iter() {
            let feat: Vec<f64> = features[idx].clone();
            let dist: f64 = feat
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - *b as f64).powi(2))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        // 3. Calculate context distribution for this state
        let mut context_counts: HashMap<i32, usize> = HashMap::new();
        for &idx in indices {
            *context_counts.entry(all_segments[idx].context).or_insert(0) += 1;
        }

        let total = indices.len() as f64;
        let (dominant_ctx, dominant_count) = context_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(&c, &cnt)| (c, cnt))
            .unwrap_or((0, 0));

        let purity = dominant_count as f64 / total;

        let context_dist: HashMap<i32, f64> = context_counts
            .iter()
            .map(|(&ctx, &count)| (ctx, count as f64 / total))
            .collect();

        // 4. Create grain entry
        let best_seg = &all_segments[best_idx];

        grain_library.push(GrainEntry {
            id: grain_library.len(),
            state_id: *state_id,
            source_file: best_seg.source_file.clone(),
            start_ms: best_seg.start_ms,
            end_ms: best_seg.end_ms,
            duration_ms: best_seg.end_ms - best_seg.start_ms,
            centroid_features: centroid,
            dominant_context: dominant_ctx,
            context_purity: purity,
            context_distribution: context_dist,
            sample_count: indices.len(),
        });
    }

    grain_library.sort_by_key(|g| std::cmp::Reverse(g.sample_count));

    println!("  Extracted {} unique grains", grain_library.len());
    println!("  Top grains by frequency:");
    for grain in grain_library.iter().take(10) {
        println!(
            "    • State {:3}: {} samples, Context {} ({:.0}% pure)",
            grain.state_id,
            grain.sample_count,
            grain.dominant_context,
            grain.context_purity * 100.0
        );
    }
    println!();

    // ---------------------------------------------------------
    // PHASE 3: MINE N-GRAMS AND BUILD SYNTACTIC TEMPLATES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/5] Mining Syntax Templates...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Build sequences per file
    let mut file_sequences: HashMap<String, (Vec<StateId>, Vec<i32>)> = HashMap::new();

    for (idx, seg) in all_segments.iter().enumerate() {
        let entry = file_sequences.entry(seg.source_file.clone()).or_default();
        entry.0.push(state_ids[idx]);
        entry.1.push(seg.context);
    }

    // Mine N-grams with context mapping
    let mut ngram_stats: HashMap<Vec<StateId>, NgramStats> = HashMap::new();

    for (_file, (sequence, contexts)) in &file_sequences {
        for n in 2..=5 {
            if sequence.len() < n {
                continue;
            }

            for i in 0..=(sequence.len() - n) {
                let ngram = sequence[i..i + n].to_vec();
                let ctx = contexts[i];

                let stats = ngram_stats.entry(ngram).or_insert(NgramStats {
                    total_count: 0,
                    context_counts: HashMap::new(),
                });

                stats.total_count += 1;
                *stats.context_counts.entry(ctx).or_insert(0) += 1;
            }
        }
    }

    // Build templates for high-purity N-grams
    let mut template_library: Vec<SyntaxTemplate> = Vec::new();
    let mut sorted_ngrams: Vec<_> = ngram_stats.iter().collect();
    sorted_ngrams.sort_by(|a, b| {
        let score_a = calculate_purity_score(a.1);
        let score_b = calculate_purity_score(b.1);
        score_b.partial_cmp(&score_a).unwrap()
    });

    // State ID -> Grain ID mapping
    let state_to_grain: HashMap<StateId, usize> = grain_library.iter().map(|g| (g.state_id, g.id)).collect();

    for (ngram, stats) in sorted_ngrams.iter() {
        if stats.total_count < 30 {
            continue;
        }

        let (dominant_ctx, dominant_count) = stats
            .context_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(&c, &cnt)| (c, cnt))
            .unwrap_or((0, 0));

        let purity = dominant_count as f64 / stats.total_count as f64;
        if purity < 0.5 {
            continue;
        }

        let context_dist: HashMap<i32, f64> = stats
            .context_counts
            .iter()
            .map(|(&ctx, &count)| (ctx, count as f64 / stats.total_count as f64))
            .collect();

        let grain_ids: Vec<usize> = ngram.iter().filter_map(|&s| state_to_grain.get(&s).copied()).collect();

        template_library.push(SyntaxTemplate {
            id: template_library.len(),
            pattern: (*ngram).clone(),
            pattern_str: format!("{:?}", ngram),
            n: ngram.len(),
            dominant_context: dominant_ctx,
            purity,
            total_occurrences: stats.total_count,
            context_distribution: context_dist,
            grain_ids,
        });

        if template_library.len() >= 500 {
            break;
        }
    }

    println!("  Created {} syntax templates", template_library.len());
    println!("  Top templates:");
    for template in template_library.iter().take(10) {
        println!(
            "    • N-{}: {} → Context {} ({:.0}% pure, {} occurrences)",
            template.n,
            template.pattern_str,
            template.dominant_context,
            template.purity * 100.0,
            template.total_occurrences
        );
    }
    println!();

    // ---------------------------------------------------------
    // PHASE 4: BUILD CONTEXT VOCABULARY
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/5] Building Context Vocabulary...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let mut context_vocab: HashMap<i32, ContextVocabulary> = HashMap::new();

    for template in &template_library {
        let entry = context_vocab
            .entry(template.dominant_context)
            .or_insert(ContextVocabulary {
                context_id: template.dominant_context,
                n_patterns: 0,
                top_patterns: Vec::new(),
                total_occurrences: 0,
            });

        entry.n_patterns += 1;
        entry.total_occurrences += template.total_occurrences;
        if entry.top_patterns.len() < 5 {
            entry.top_patterns.push(template.pattern_str.clone());
        }
    }

    println!("  Context vocabulary:");
    let mut sorted_vocab: Vec<_> = context_vocab.iter().collect();
    sorted_vocab.sort_by_key(|(_, v)| std::cmp::Reverse(v.n_patterns));

    for (ctx, vocab) in sorted_vocab.iter() {
        println!(
            "    • Context {}: {} patterns, {} total occurrences",
            ctx, vocab.n_patterns, vocab.total_occurrences
        );
    }
    println!();

    // ---------------------------------------------------------
    // PHASE 5: SAVE MANIFEST
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[5/5] Saving Library Manifest...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let manifest = LibraryManifest {
        total_segments,
        total_files: unique_files.len(),
        n_acoustic_states: n_states,
        n_grains: grain_library.len(),
        n_templates: template_library.len(),
        discretization_bins: bins_per_dim,
        created: chrono::Local::now().to_rfc3339(),
        grains: grain_library,
        templates: template_library,
        context_vocabulary: context_vocab,
    };

    let manifest_path = output_dir.join("library_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&manifest_path, manifest_json)?;

    println!("  ✓ Manifest saved to: {}", manifest_path.display());
    println!();

    // Summary
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  ROSETTA SYNTHESIS LIBRARY SUMMARY                                       │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Source segments:     {:>8}                                    ",
        total_segments
    );
    println!(
        "  │  Source files:        {:>8}                                    ",
        unique_files.len()
    );
    println!(
        "  │  Acoustic states:     {:>8}                                    ",
        n_states
    );
    println!(
        "  │  Grains extracted:    {:>8}                                    ",
        manifest.n_grains
    );
    println!(
        "  │  Syntax templates:    {:>8}                                    ",
        manifest.n_templates
    );
    println!(
        "  │  Behavioral contexts: {:>8}                                    ",
        manifest.context_vocabulary.len()
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Usage example
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  HOW TO USE THE ROSETTA SYNTH                                           │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │                                                                          │");
    println!("  │  1. IDENTIFY INTENT:                                                     │");
    println!("  │     \"I need to signal Territorial (Context 6)\"                          │");
    println!("  │                                                                          │");
    println!("  │  2. RETRIEVE TEMPLATE:                                                   │");
    println!("  │     Look up Context 6 → Template [391, 391, 391, 391]                   │");
    println!("  │                                                                          │");
    println!("  │  3. SYNTHESIZE:                                                          │");
    println!("  │     For each state in template:                                          │");
    println!("  │       - Load grain audio from source file                               │");
    println!("  │       - Extract segment [start_ms, end_ms]                              │");
    println!("  │       - Concatenate grains                                               │");
    println!("  │       - Apply prosody (pitch/rate adjustment)                           │");
    println!("  │                                                                          │");
    println!("  │  4. PLAY                                                                  │");
    println!("  │                                                                          │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

fn calculate_purity_score(stats: &NgramStats) -> f64 {
    if stats.total_count == 0 {
        return 0.0;
    }
    let max_count = stats.context_counts.values().max().copied().unwrap_or(0);
    let purity = max_count as f64 / stats.total_count as f64;
    (stats.total_count as f64) * purity * purity
}
