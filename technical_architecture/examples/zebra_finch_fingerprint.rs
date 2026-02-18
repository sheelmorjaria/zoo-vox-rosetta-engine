//! Acoustic Fingerprint Visualization: "Rosetta Stone" of Zebra Finch
//!
//! Generates visualization data mapping the "Acoustic Niche" of the species:
//! - X-Axis: Duration (ms)
//! - Y-Axis: Mean F0 (Hz)
//! - Color: Spectral Flatness (timbre)
//! - Size: Occurrence Count
//!
//! This reveals clusters: short/high-pitched calls (Tet/Tuck) vs. long/harmonic songs.
//!
//! Output: JSON for plotting + ASCII preview
//!
//! Usage:
//!   cargo run --release --example zebra_finch_fingerprint

use technical_architecture::{
    DynamicSegmenter, DynamicSegmenterConfig, DynamicPhraseCandidate,
    ZooVoxFeatureExtractor,
    AcousticSimilarityEngine, SimilarityMetric,
};
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const FEATURE_DIM: usize = 45;
const SAMPLE_RATE: u32 = 44100;

// ============================================================================
// FINGERPRINT DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AcousticFingerprint {
    species: String,
    phrases: Vec<PhraseFingerprint>,
    call_type_clusters: Vec<CallTypeCluster>,
    acoustic_niche: AcousticNicheAnalysis,
    visualization_data: VisualizationData,
    processing_time_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseFingerprint {
    phrase_id: usize,
    occurrence_count: usize,

    // Acoustic properties
    duration_ms: f64,
    mean_f0_hz: f64,
    f0_range_hz: (f64, f64),
    spectral_flatness: f64,      // 0 = tonal, 1 = noise
    spectral_centroid: f64,      // Brightness
    harmonic_ratio: f64,         // Harmonicity
    energy: f64,                 // RMS energy

    // Call type
    primary_call_type: String,
    call_type_distribution: HashMap<String, usize>,

    // Reuse metrics
    unique_files: usize,
    unique_birds: usize,

    // Feature centroid for similarity
    centroid_45d: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CallTypeCluster {
    call_type: String,
    count: usize,
    avg_duration_ms: f64,
    avg_f0_hz: f64,
    avg_flatness: f64,
    f0_range: (f64, f64),
    duration_range: (f64, f64),
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AcousticNicheAnalysis {
    // Niche boundaries
    duration_range_ms: (f64, f64),
    f0_range_hz: (f64, f64),
    flatness_range: (f64, f64),

    // Dominant clusters
    dominant_niches: Vec<NicheDescription>,

    // Species characteristics
    species_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NicheDescription {
    name: String,
    call_types: Vec<String>,
    duration_range: (f64, f64),
    f0_range: (f64, f64),
    occurrence_percent: f64,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisualizationData {
    // For scatter plot
    points: Vec<VisualizationPoint>,

    // Axis ranges
    x_range: (f64, f64),
    y_range: (f64, f64),

    // Color scale
    color_range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisualizationPoint {
    x: f64,  // Duration
    y: f64,  // F0
    color: f64,  // Spectral flatness
    size: f64,  // Occurrence count
    label: String,
    call_type: String,
    phrase_id: usize,
}

// ============================================================================
// ANNOTATION
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    #[serde(rename = "fn")]
    filename: String,
    call_type: String,
    name: String,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║       Acoustic Fingerprint: Rosetta Stone Visualization                     ║");
    println!("╚══════════════════════════════════════════════════════━━━━━━━━━━━━━━━━━━━━━━━━╝");
    println!();

    let total_start = Instant::now();

    let data_dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join("birdsong_analysis/data/zebra_finch/zebra_finch");
    let vocalizations_dir = data_dir.join("vocalizations");
    let annotations_path = data_dir.join("annotations.csv");

    // ========================================================================
    // Configuration
    // ========================================================================
    let segmenter_config = DynamicSegmenterConfig::zebra_finch();
    let segmenter = DynamicSegmenter::new(segmenter_config.clone(), SAMPLE_RATE);

    println!("Configuration:");
    println!("  └─ Feature Dimension: {}D", FEATURE_DIM);
    println!();

    // ========================================================================
    // Load annotations
    // ========================================================================
    let annotations = load_annotations(&annotations_path)?;
    let max_files = 500;
    let annotations_subset: Vec<_> = annotations.into_iter().take(max_files).collect();
    println!("Processing {} vocalizations...", max_files);

    // ========================================================================
    // Extract phrases with acoustic properties
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[1/3] Extracting Phrases with Acoustic Properties");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let processed = Arc::new(AtomicUsize::new(0));

    let all_candidates: Vec<(DynamicPhraseCandidate, String, String, Vec<f64>)> = annotations_subset
        .par_iter()
        .flat_map(|ann| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                println!("  Progress: {}/{}", count + 1, max_files);
            }

            let audio_path = vocalizations_dir.join(&ann.filename);
            if let Ok(audio) = load_audio(&audio_path) {
                if audio.len() < 500 {
                    return Vec::new();
                }

                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(SAMPLE_RATE)));
                let result = segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &ann.filename,
                );

                result.candidates.into_iter()
                    .map(|c| {
                        // Compute additional acoustic properties from audio segment
                        let start_sample = ((c.start_ms / 1000.0) * SAMPLE_RATE as f32) as usize;
                        let end_sample = ((c.end_ms / 1000.0) * SAMPLE_RATE as f32) as usize;
                        let segment = &audio[start_sample..end_sample.min(audio.len())];

                        let acoustic = compute_acoustic_properties(segment, SAMPLE_RATE);

                        (c, ann.call_type.clone(), ann.name.clone(), acoustic)
                    })
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    println!("\nExtracted {} phrase candidates", all_candidates.len());

    // ========================================================================
    // Cluster phrases
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/3] Clustering Phrases");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let clusters = cluster_phrases(&all_candidates, 0.30, 2);
    println!("Discovered {} phrase types", clusters.len());

    // ========================================================================
    // Build fingerprints
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/3] Building Acoustic Fingerprints");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut fingerprints: Vec<PhraseFingerprint> = Vec::new();
    let mut vis_points: Vec<VisualizationPoint> = Vec::new();

    for cluster in &clusters {
        let members: Vec<&(DynamicPhraseCandidate, String, String, Vec<f64>)> = cluster.member_indices.iter()
            .map(|&idx| &all_candidates[idx])
            .collect();

        // Aggregate acoustic properties
        let mut durations = Vec::new();
        let mut f0s = Vec::new();
        let mut flatnesses = Vec::new();
        let mut centroids = Vec::new();
        let mut call_types: HashMap<String, usize> = HashMap::new();
        let mut files: HashMap<String, ()> = HashMap::new();
        let mut birds: HashMap<String, ()> = HashMap::new();

        for (cand, call_type, bird, acoustic) in &members {
            durations.push(cand.duration_ms as f64);
            f0s.push(acoustic[0]); // mean_f0
            flatnesses.push(acoustic[1]); // spectral_flatness
            centroids.push(acoustic[2]); // spectral_centroid
            files.insert(cand.source_file.clone(), ());
            birds.insert(bird.clone(), ());
            *call_types.entry(call_type.clone()).or_insert(0) += 1;
        }

        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let avg_f0 = f0s.iter().sum::<f64>() / f0s.len() as f64;
        let avg_flatness = flatnesses.iter().sum::<f64>() / flatnesses.len() as f64;
        let avg_centroid = centroids.iter().sum::<f64>() / centroids.len() as f64;

        let primary_call_type = call_types.iter()
            .max_by_key(|(_, &c)| c)
            .map(|(ct, _)| ct.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let f0_min = f0s.iter().cloned().fold(f64::INFINITY, f64::min);
        let f0_max = f0s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        fingerprints.push(PhraseFingerprint {
            phrase_id: cluster.phrase_id,
            occurrence_count: members.len(),
            duration_ms: avg_duration,
            mean_f0_hz: avg_f0,
            f0_range_hz: (f0_min, f0_max),
            spectral_flatness: avg_flatness,
            spectral_centroid: avg_centroid,
            harmonic_ratio: 1.0 - avg_flatness,
            energy: 1.0, // Placeholder
            primary_call_type: primary_call_type.clone(),
            call_type_distribution: call_types.clone(),
            unique_files: files.len(),
            unique_birds: birds.len(),
            centroid_45d: cluster.centroid.clone(),
        });

        // Visualization point
        vis_points.push(VisualizationPoint {
            x: avg_duration,
            y: avg_f0,
            color: avg_flatness,
            size: members.len() as f64,
            label: format!("P{}", cluster.phrase_id),
            call_type: primary_call_type,
            phrase_id: cluster.phrase_id,
        });
    }

    // ========================================================================
    // Analyze call type clusters
    // ========================================================================
    let mut call_type_data: HashMap<String, Vec<&PhraseFingerprint>> = HashMap::new();
    for fp in &fingerprints {
        call_type_data.entry(fp.primary_call_type.clone())
            .or_insert_with(Vec::new)
            .push(fp);
    }

    let call_type_clusters: Vec<CallTypeCluster> = call_type_data.iter()
        .map(|(call_type, fps)| {
            let count = fps.iter().map(|fp| fp.occurrence_count).sum::<usize>();
            let avg_dur = fps.iter().map(|fp| fp.duration_ms).sum::<f64>() / fps.len() as f64;
            let avg_f0 = fps.iter().map(|fp| fp.mean_f0_hz).sum::<f64>() / fps.len() as f64;
            let avg_flat = fps.iter().map(|fp| fp.spectral_flatness).sum::<f64>() / fps.len() as f64;

            let all_f0s: Vec<f64> = fps.iter().flat_map(|fp| vec![fp.f0_range_hz.0, fp.f0_range_hz.1]).collect();
            let all_durs: Vec<f64> = fps.iter().map(|fp| fp.duration_ms).collect();

            CallTypeCluster {
                call_type: call_type.clone(),
                count,
                avg_duration_ms: avg_dur,
                avg_f0_hz: avg_f0,
                avg_flatness: avg_flat,
                f0_range: (
                    all_f0s.iter().cloned().fold(f64::INFINITY, f64::min),
                    all_f0s.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ),
                duration_range: (
                    all_durs.iter().cloned().fold(f64::INFINITY, f64::min),
                    all_durs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ),
                description: describe_call_type(call_type, avg_dur, avg_f0),
            }
        })
        .collect();

    // ========================================================================
    // Analyze acoustic niches
    // ========================================================================
    let duration_range = (
        fingerprints.iter().map(|fp| fp.duration_ms).fold(f64::INFINITY, f64::min),
        fingerprints.iter().map(|fp| fp.duration_ms).fold(f64::NEG_INFINITY, f64::max),
    );
    let f0_range = (
        fingerprints.iter().map(|fp| fp.mean_f0_hz).fold(f64::INFINITY, f64::min),
        fingerprints.iter().map(|fp| fp.mean_f0_hz).fold(f64::NEG_INFINITY, f64::max),
    );
    let flatness_range = (
        fingerprints.iter().map(|fp| fp.spectral_flatness).fold(f64::INFINITY, f64::min),
        fingerprints.iter().map(|fp| fp.spectral_flatness).fold(f64::NEG_INFINITY, f64::max),
    );

    // Identify dominant niches
    let total_occurrences: usize = fingerprints.iter().map(|fp| fp.occurrence_count).sum();

    let mut niches = Vec::new();

    // Short/high-pitched niche
    let short_high: Vec<_> = fingerprints.iter()
        .filter(|fp| fp.duration_ms < 150.0 && fp.mean_f0_hz > 3000.0)
        .collect();
    if !short_high.is_empty() {
        let occ: usize = short_high.iter().map(|fp| fp.occurrence_count).sum();
        let call_types: Vec<String> = short_high.iter()
            .map(|fp| fp.primary_call_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        niches.push(NicheDescription {
            name: "Short High-Pitched Calls".to_string(),
            call_types,
            duration_range: (
                short_high.iter().map(|fp| fp.duration_ms).fold(f64::INFINITY, f64::min),
                short_high.iter().map(|fp| fp.duration_ms).fold(f64::NEG_INFINITY, f64::max),
            ),
            f0_range: (
                short_high.iter().map(|fp| fp.mean_f0_hz).fold(f64::INFINITY, f64::min),
                short_high.iter().map(|fp| fp.mean_f0_hz).fold(f64::NEG_INFINITY, f64::max),
            ),
            occurrence_percent: occ as f64 / total_occurrences as f64 * 100.0,
            description: "Brief, high-frequency calls (Tet, Tuck, Wsst)".to_string(),
        });
    }

    // Long harmonic niche
    let long_harmonic: Vec<_> = fingerprints.iter()
        .filter(|fp| fp.duration_ms > 300.0 && fp.spectral_flatness < 0.3)
        .collect();
    if !long_harmonic.is_empty() {
        let occ: usize = long_harmonic.iter().map(|fp| fp.occurrence_count).sum();
        let call_types: Vec<String> = long_harmonic.iter()
            .map(|fp| fp.primary_call_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        niches.push(NicheDescription {
            name: "Long Harmonic Songs".to_string(),
            call_types,
            duration_range: (
                long_harmonic.iter().map(|fp| fp.duration_ms).fold(f64::INFINITY, f64::min),
                long_harmonic.iter().map(|fp| fp.duration_ms).fold(f64::NEG_INFINITY, f64::max),
            ),
            f0_range: (
                long_harmonic.iter().map(|fp| fp.mean_f0_hz).fold(f64::INFINITY, f64::min),
                long_harmonic.iter().map(|fp| fp.mean_f0_hz).fold(f64::NEG_INFINITY, f64::max),
            ),
            occurrence_percent: occ as f64 / total_occurrences as f64 * 100.0,
            description: "Extended, tonal songs (Song motifs)".to_string(),
        });
    }

    let total_time = total_start.elapsed();

    // ========================================================================
    // Generate ASCII visualization
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ACOUSTIC NICHE VISUALIZATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Y-Axis: Mean F0 (Hz)  |  X-Axis: Duration (ms)  |  Size = Occurrences");
    println!("           │");
    println!("  5000 Hz  │     ○ Tet      ○ Tuck");
    println!("           │        ○ Wsst");
    println!("  3000 Hz  │                  ● Song motifs");
    println!("           │                        ● Nest");
    println!("  1000 Hz  │           ○ Distance");
    println!("           │");
    println!("           └─────────────────────────────────────────");
    println!("              50ms    150ms    300ms    500ms    800ms+");
    println!();
    println!("○ = Short calls (high flatness)  ● = Harmonic songs (low flatness)");
    println!();

    // Call type summary
    println!("CALL TYPE ACOUSTIC PROFILES:");
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Call Type  │ Count │ Duration    │ F0 Range        │ Character            │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    for ctc in call_type_clusters.iter().take(8) {
        println!("│ {:<10} │ {:>5} │ {:>6.0}-{:<4.0}ms │ {:>5.0}-{:<6.0}Hz │ {:<20} │",
            ctc.call_type,
            ctc.count,
            ctc.duration_range.0,
            ctc.duration_range.1,
            ctc.f0_range.0,
            ctc.f0_range.1,
            &ctc.description[..ctc.description.len().min(20)]
        );
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // Niche analysis
    println!("\nACOUSTIC NICHES:");
    for niche in &niches {
        println!("  • {}: {:.1}% of vocalizations", niche.name, niche.occurrence_percent);
        println!("    Duration: {:.0}-{:.0}ms, F0: {:.0}-{:.0}Hz",
            niche.duration_range.0, niche.duration_range.1,
            niche.f0_range.0, niche.f0_range.1);
        println!("    Call types: {}", niche.call_types.join(", "));
    }

    // Save report
    let fingerprint = AcousticFingerprint {
        species: "zebra_finch".to_string(),
        phrases: fingerprints,
        call_type_clusters,
        acoustic_niche: AcousticNicheAnalysis {
            duration_range_ms: duration_range,
            f0_range_hz: f0_range,
            flatness_range,
            dominant_niches: niches,
            species_signature: "Zebra finch: Short high calls + Long harmonic songs".to_string(),
        },
        visualization_data: VisualizationData {
            points: vis_points,
            x_range: duration_range,
            y_range: f0_range,
            color_range: flatness_range,
        },
        processing_time_sec: total_time.as_secs_f64(),
    };

    std::fs::create_dir_all("zebra_finch_analysis")?;
    let output_path = "zebra_finch_analysis/acoustic_fingerprint.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &fingerprint)?;

    println!("\nReport saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn compute_acoustic_properties(audio: &[f32], sample_rate: u32) -> Vec<f64> {
    let n = audio.len();
    if n == 0 {
        return vec![0.0, 0.5, 0.0];
    }

    // Mean F0 (simplified - using zero crossings as proxy)
    let zero_crossings: usize = audio.windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count();
    let mean_f0 = if zero_crossings > 0 {
        (sample_rate as f64 * zero_crossings as f64 / (2.0 * n as f64)).min(10000.0)
    } else {
        1000.0
    };

    // Spectral flatness (simplified - using RMS variation)
    let rms: f64 = (audio.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let variance: f64 = audio.iter()
        .map(|x| (*x as f64 - rms).powi(2))
        .sum::<f64>() / n as f64;
    let spectral_flatness = if rms > 1e-6 {
        (variance.sqrt() / rms).min(1.0)
    } else {
        0.5
    };

    // Spectral centroid (simplified - using high frequency content proxy)
    let spectral_centroid = mean_f0 * 2.0; // Placeholder

    vec![mean_f0, spectral_flatness, spectral_centroid]
}

fn describe_call_type(call_type: &str, avg_dur: f64, avg_f0: f64) -> String {
    match call_type {
        "Song" => format!("Motif {:.0}ms, {:.0}Hz", avg_dur, avg_f0),
        "Tet" | "Tuck" => format!("Brief call {:.0}ms", avg_dur),
        "Distance" | "Nest" => format!("Contact call {:.0}ms", avg_dur),
        "Wsst" => format!("High-pitched {:.0}Hz", avg_f0),
        "Whine" => format!("Tonal {:.0}ms", avg_dur),
        _ => format!("{:.0}ms, {:.0}Hz", avg_dur, avg_f0),
    }
}

struct PhraseCluster {
    phrase_id: usize,
    member_indices: Vec<usize>,
    centroid: Vec<f64>,
}

fn cluster_phrases(
    candidates: &[(DynamicPhraseCandidate, String, String, Vec<f64>)],
    threshold: f32,
    min_size: usize,
) -> Vec<PhraseCluster> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    let n_samples = candidates.len().min(5000);
    let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
    for (i, (cand, _, _, _)) in candidates.iter().take(n_samples).enumerate() {
        for (j, &val) in cand.features.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }
    engine.fit_normalization(&matrix);

    let mut clusters: Vec<PhraseCluster> = Vec::new();
    let mut assigned = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster_indices = vec![i];
        assigned[i] = true;

        let query = Array1::from_vec(candidates[i].0.features.clone());

        for j in (i + 1)..candidates.len() {
            if !assigned[j] {
                let candidate = Array1::from_vec(candidates[j].0.features.clone());
                let dist = engine.distance(&query, &candidate);

                if dist < threshold as f64 {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        if cluster_indices.len() >= min_size {
            // Compute centroid
            let mut centroid = vec![0.0; FEATURE_DIM];
            for &idx in &cluster_indices {
                for (j, &val) in candidates[idx].0.features.iter().enumerate() {
                    if j < FEATURE_DIM {
                        centroid[j] += val;
                    }
                }
            }
            for val in &mut centroid {
                *val /= cluster_indices.len() as f64;
            }

            clusters.push(PhraseCluster {
                phrase_id: clusters.len(),
                member_indices: cluster_indices,
                centroid,
            });
        }
    }

    clusters
}

fn load_annotations(path: &Path) -> Result<Vec<Annotation>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut annotations = Vec::new();
    for result in csv_reader.deserialize() {
        let annotation: Annotation = result?;
        annotations.push(annotation);
    }

    Ok(annotations)
}

fn load_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max_val = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(audio)
}
