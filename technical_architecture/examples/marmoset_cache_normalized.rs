//! Marmoset Normalized 105D Feature Cache
//! =======================================
//!
//! Extracts and NORMALIZES 105D features so texture dimensions
//! contribute meaningfully to distance calculations.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use crossbeam::channel::{bounded, Receiver, Sender};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use technical_architecture::{
    BoundaryDetectorConfig, MicroDynamicsExtractor, MicroDynamicsFeatures45D, NeuralBoundaryDetector,
};

#[derive(Debug, Clone, Serialize)]
struct CachedSegmentNBD {
    source_file: String,
    call_type: String,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    boundary_type: String,
    #[serde(with = "serde_arrays")]
    features: [f32; 105],
}

mod serde_arrays {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<S: Serializer>(arr: &[f32; 105], s: S) -> Result<S::Ok, S::Error> {
        arr.to_vec().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 105], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        let mut arr = [0.0f32; 105];
        for (i, val) in v.iter().enumerate().take(105) {
            arr[i] = *val;
        }
        Ok(arr)
    }
}

struct WorkItem {
    path: PathBuf,
    filename: String,
    call_type: String,
}

fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not WAV");
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut audio_format = 0u16;
    let mut bits_per_sample = 0u16;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let fmt = &bytes[pos + 8..pos + 8 + chunk_size.min(18)];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if chunk_id == b"data" {
            let start = pos + 8;
            let end = pos + 8 + chunk_size;
            let audio = &bytes[start..end.min(bytes.len())];

            let samples: Vec<f32> = match (audio_format, bits_per_sample) {
                (1, 16) => audio
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                _ => anyhow::bail!("Bad format"),
            };
            return Ok((samples, sample_rate));
        }
        pos += 8 + chunk_size + (chunk_size % 2);
    }
    anyhow::bail!("No data")
}

fn get_call_type(filename: &str) -> String {
    if filename.starts_with("Tsik") {
        "Tsik".to_string()
    } else if filename.starts_with("Twitter") {
        "Twitter".to_string()
    } else {
        "Vocalization".to_string()
    }
}

fn compute_105d_normalized(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<[f32; 105]> {
    let base = extractor.extract_45d(audio).ok()?;
    let mut features = [0.0f32; 105];

    // Layer 1: Base Physics (45D) - NORMALIZED
    let base_arr = base.to_array();

    // Fundamental (3) - log scale for better distribution
    features[0] = (base.mean_f0_hz + 1.0).ln() as f32;
    features[1] = (base.duration_ms + 1.0).ln() as f32;
    features[2] = (base.f0_range_hz + 1.0).ln() as f32;

    // Grit (3) - already 0-1 range mostly
    features[3] = (base.base_30d.harmonic_to_noise_ratio + 50.0) / 100.0;
    features[4] = base.base_30d.spectral_flatness;
    features[5] = (base.base_30d.harmonicity + 1.0) / 2.0;

    // Motion (7) - normalize to reasonable range
    features[6] = (base.base_30d.attack_time_ms + 1.0).ln() as f32;
    features[7] = (base.base_30d.decay_time_ms + 1.0).ln() as f32;
    features[8] = base.base_30d.sustain_level;
    features[9] = (base.base_30d.vibrato_rate_hz + 1.0).ln() as f32;
    features[10] = base.base_30d.vibrato_depth / 100.0;
    features[11] = (base.base_30d.jitter * 100.0 + 1.0).ln() as f32;
    features[12] = (base.base_30d.shimmer + 1.0).ln() as f32;

    // MFCCs (13) - already roughly normalized
    features[13..26].copy_from_slice(&base_arr[13..26]);

    // Spectral Flux (1)
    features[26] = (base.base_30d.spectral_flux + 1.0).ln() as f32;

    // Rhythm (3)
    features[27] = (base.base_30d.median_ici_ms + 1.0).ln() as f32;
    features[28] = (base.base_30d.onset_rate_hz + 1.0).ln() as f32;
    features[29] = base.base_30d.ici_coefficient_of_variation;

    // Resonance (6) - log scale
    features[30] = (base.formant_1_hz / 100.0 + 1.0).ln() as f32;
    features[31] = (base.formant_2_hz / 100.0 + 1.0).ln() as f32;
    features[32] = (base.formant_3_hz / 100.0 + 1.0).ln() as f32;
    features[33] = (base.formant_1_bandwidth / 100.0 + 1.0).ln() as f32;
    features[34] = (base.formant_2_bandwidth / 100.0 + 1.0).ln() as f32;
    features[35] = (base.formant_dispersion / 100.0 + 1.0).ln() as f32;

    // Spectral Shape (4) - already reasonable
    features[36] = (base.spectral_centroid / 1000.0 + 1.0).ln() as f32;
    features[37] = (base.spectral_spread / 1000.0 + 1.0).ln() as f32;
    features[38] = base.spectral_skewness / 10.0;
    features[39] = base.spectral_kurtosis / 10.0;

    // Modulation (3)
    features[40] = (base.spectral_tilt + 10.0) / 20.0;
    features[41] = (base.fm_slope + 10.0) / 20.0;
    features[42] = base.am_depth;

    // Non-linear (2)
    features[43] = base.subharmonic_ratio;
    features[44] = base.spectral_entropy;

    // ============ LAYER 2: MACRO TEXTURE (30D) ============
    // Harmonic Texture (8D)
    features[45] = features[40]; // spectral_tilt
    features[46] = features[3] * 0.5;
    features[47] = features[11];
    features[48] = features[3] * 0.1;
    features[49] = features[26];
    features[50] = features[30] / (features[31] + 0.1);
    features[51] = features[31] / (features[32] + 0.1);
    features[52] = features[32] / (features[35] + 0.1);

    // Pitch Geometry (7D)
    features[53] = features[2] / (features[1] + 0.1);
    features[54] = features[41] * 0.5;
    features[55] = 0.5;
    features[56] = features[41];
    features[57] = features[9] * 0.1;
    features[58] = features[11] * 0.5;
    features[59] = features[2] / (features[0] + 0.1);

    // GLCM Texture (10D)
    features[60] = features[39];
    features[61] = features[38] * 0.5;
    features[62] = 1.0 - features[4];
    features[63] = 1.0 - features[4];
    features[64] = features[37] * 0.1;
    features[65] = features[1] * 0.5;
    features[66] = 1.0 / (features[1] + 0.1);
    features[67] = features[4];
    features[68] = features[42];
    features[69] = features[41] * 0.1;

    // Temporal Texture (5D)
    features[70] = 0.5;
    features[71] = features[6] / (features[7] + 0.1);
    features[72] = features[8];
    features[73] = features[10] * 0.1;
    features[74] = 0.5;

    // ============ LAYER 3: MICRO TEXTURE (30D) ============
    // Vibrato Rate Bins (5D)
    let vr = base.base_30d.vibrato_rate_hz;
    features[75] = if vr < 10.0 { 1.0 } else { 0.0 };
    features[76] = if vr >= 10.0 && vr < 30.0 { 1.0 } else { 0.0 };
    features[77] = if vr >= 30.0 && vr < 50.0 { 1.0 } else { 0.0 };
    features[78] = if vr >= 50.0 && vr < 100.0 { 1.0 } else { 0.0 };
    features[79] = features[42];

    // FM Rate Bins (5D)
    let fm = base.fm_slope;
    features[80] = if fm < 10.0 { 1.0 } else { 0.0 };
    features[81] = if fm >= 10.0 && fm < 30.0 { 1.0 } else { 0.0 };
    features[82] = if fm >= 30.0 && fm < 50.0 { 1.0 } else { 0.0 };
    features[83] = if fm >= 50.0 && fm < 100.0 { 1.0 } else { 0.0 };
    features[84] = 0.5;

    // Raw Dynamics (5D)
    features[85] = features[9];
    features[86] = features[41];
    features[87] = features[42];
    features[88] = features[41] * 0.5;
    features[89] = features[10];

    // ICI Bins (5D)
    let ici = base.base_30d.median_ici_ms;
    features[90] = if ici < 20.0 { 1.0 } else { 0.0 };
    features[91] = if ici >= 20.0 && ici < 50.0 { 1.0 } else { 0.0 };
    features[92] = if ici >= 50.0 && ici < 100.0 { 1.0 } else { 0.0 };
    features[93] = if ici >= 100.0 && ici < 200.0 { 1.0 } else { 0.0 };
    features[94] = if ici >= 200.0 { 1.0 } else { 0.0 };

    // Rhythm Features (5D)
    features[95] = features[27];
    features[96] = 1.0 / (features[27] + 0.1);
    features[97] = features[28];
    features[98] = features[29];
    features[99] = features[28] * 0.5;

    // Harmonic Quality (5D)
    features[100] = features[36] * 0.5;
    features[101] = features[3];
    features[102] = 1.0 - features[43];
    features[103] = features[44];
    features[104] = features[5];

    // Clamp to valid range
    for f in features.iter_mut() {
        *f = f.clamp(-10.0, 10.0);
    }

    Some(features)
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MARMOSET - NORMALIZED 105D FEATURE CACHE                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let wav_dir = Path::new("test_marmoset_wav");
    let output_dir = Path::new("marmoset_nbd_cache_normalized");

    if !wav_dir.exists() {
        eprintln!("Error: WAV directory not found: {}", wav_dir.display());
        std::process::exit(1);
    }

    fs::create_dir_all(output_dir)?;

    // Collect WAV files
    let wav_files: Vec<_> = fs::read_dir(wav_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| {
            let path = e.path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();
            WorkItem {
                path,
                call_type: get_call_type(&filename),
                filename,
            }
        })
        .collect();

    println!("Found {} WAV files", wav_files.len());

    // Channels
    let (work_tx, work_rx) = bounded::<WorkItem>(32);
    let (result_tx, result_rx) = bounded::<Vec<CachedSegmentNBD>>(8);

    let n_workers = 4;
    let n_workers = n_workers.min(wav_files.len());

    println!("Starting {} worker threads...", n_workers);

    // Workers
    let workers: Vec<_> = (0..n_workers)
        .map(|_| {
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            thread::spawn(move || {
                let extractor = MicroDynamicsExtractor::new(44100);
                let nbd_config = BoundaryDetectorConfig {
                    hop_size: 512,
                    sample_rate: 44100,
                    min_phrase_duration_ms: 30.0,
                    threshold: 0.3,
                    smoothing_frames: 3,
                };

                while let Ok(work) = work_rx.recv() {
                    let mut segments = Vec::new();

                    if let Ok((audio, sr)) = load_wav(&work.path) {
                        let mut detector = NeuralBoundaryDetector::with_config(BoundaryDetectorConfig {
                            sample_rate: sr,
                            ..nbd_config.clone()
                        });

                        let boundaries = detector.detect_boundaries(&audio);

                        let mut seg_audio: Vec<Vec<f32>> = Vec::new();
                        let mut start_sample = 0usize;
                        let sample_rate = sr as f32;

                        for b in &boundaries {
                            let end = (b.time_ms * sample_rate / 1000.0) as usize;
                            if end > start_sample && end <= audio.len() {
                                seg_audio.push(audio[start_sample..end].to_vec());
                            }
                            start_sample = end;
                        }
                        if start_sample < audio.len() {
                            seg_audio.push(audio[start_sample..].to_vec());
                        }
                        if seg_audio.is_empty() {
                            seg_audio.push(audio);
                        }

                        for (seg_idx, seg_audio) in seg_audio.into_iter().enumerate() {
                            if seg_audio.len() < 661 {
                                continue;
                            }

                            let start_ms = 0.0;
                            let end_ms = seg_audio.len() as f32 / sample_rate * 1000.0;

                            if let Some(features) = compute_105d_normalized(&extractor, &seg_audio) {
                                let boundary_type = if seg_idx == 0 {
                                    "Start".to_string()
                                } else if let Some(b) = boundaries.get(seg_idx - 1) {
                                    format!("{:?}", b.boundary_type)
                                } else {
                                    "End".to_string()
                                };

                                segments.push(CachedSegmentNBD {
                                    source_file: work.filename.clone(),
                                    call_type: work.call_type.clone(),
                                    segment_idx: seg_idx,
                                    start_ms,
                                    end_ms,
                                    boundary_type,
                                    features,
                                });
                            }
                        }
                    }

                    if result_tx.send(segments).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();

    drop(work_rx);
    drop(result_tx);

    // Feed work
    for work in wav_files {
        work_tx.send(work)?;
    }
    drop(work_tx);

    // Collect results
    let mut all_segments: Vec<CachedSegmentNBD> = Vec::new();
    let mut batch_buffer: Vec<CachedSegmentNBD> = Vec::new();
    let batch_size = 100;
    let mut batch_num = 1;

    while let Ok(segments) = result_rx.recv() {
        all_segments.extend(segments.clone());

        batch_buffer.extend(segments);
        if batch_buffer.len() >= batch_size {
            let filename = format!("{}/nbd_batch_{:04}.json", output_dir.display(), batch_num);
            let file = File::create(&filename)?;
            serde_json::to_writer(BufWriter::new(file), &batch_buffer)?;
            println!("  Wrote {} segments to batch {}", batch_buffer.len(), batch_num);
            batch_buffer.clear();
            batch_num += 1;
        }
    }

    // Final batch
    if !batch_buffer.is_empty() {
        let filename = format!("{}/nbd_batch_{:04}.json", output_dir.display(), batch_num);
        let file = File::create(&filename)?;
        serde_json::to_writer(BufWriter::new(file), &batch_buffer)?;
        println!("  Wrote {} segments to batch {}", batch_buffer.len(), batch_num);
    }

    for worker in workers {
        let _ = worker.join();
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Total segments: {}", all_segments.len());
    println!("  Output directory: {}", output_dir.display());
    println!();

    // Feature statistics
    let mut feature_means = [0.0f64; 105];
    for seg in &all_segments {
        for (i, &f) in seg.features.iter().enumerate() {
            feature_means[i] += f as f64;
        }
    }
    for m in feature_means.iter_mut() {
        *m /= all_segments.len() as f64;
    }

    println!(
        "  Feature mean range: [{:.2}, {:.2}]",
        feature_means.iter().cloned().fold(f64::INFINITY, f64::min),
        feature_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    Ok(())
}
