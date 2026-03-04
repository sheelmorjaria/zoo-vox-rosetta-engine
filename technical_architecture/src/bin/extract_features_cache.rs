//! Pre-extract and cache 105D features for all BEANS-Zero samples
//!
//! This dramatically speeds up evaluation by caching features to disk.
//!
//! Supports multiple audio formats:
//!   - WAV via hound
//!   - FLAC, MP3, AAC, OGG via symphonia
//!   - Raw float32 (.raw)
//!
//! Usage:
//!   cargo run --release --bin extract_features_cache -- beans_zero_cache/beans_audio_manifest.json

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use technical_architecture::{MicroDynamicsExtractor, MicroDynamicsFeatures};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansSample {
    audio_file: String,
    n_samples: u32,
    sample_rate: Option<u32>,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: Option<String>,
    dataset_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFeatures {
    features: Vec<f32>,
    audio_file: String,
}

// ============================================================================
// Audio Loading - Multi-format Support
// ============================================================================

/// Load audio from file, auto-detecting format from extension
fn load_audio_file(path: &Path, expected_samples: Option<u32>) -> Result<(Vec<f32>, u32)> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "raw" => {
            // Raw float32 PCM
            let expected = expected_samples.context("Raw audio requires expected_samples")?;
            let audio = load_raw_audio(path, expected)?;
            let sr = 44100; // Default for raw
            Ok((audio, sr))
        }
        "wav" | "wave" => {
            // WAV via hound
            load_wav_audio(path)
        }
        "flac" | "mp3" | "aac" | "ogg" | "m4a" | "mp4" => {
            // Symphonia for other formats
            #[cfg(feature = "symphonia")]
            {
                load_symphonia_audio(path)
            }
            #[cfg(not(feature = "symphonia"))]
            {
                bail!("Symphonia feature not enabled. Add --features symphonia to build.")
            }
        }
        _ => {
            bail!("Unsupported audio format: {}", extension)
        }
    }
}

/// Load raw float32 PCM audio
fn load_raw_audio(path: &Path, expected_samples: u32) -> Result<Vec<f32>> {
    let mut file = fs::File::open(path)?;
    let mut buffer = Vec::new();
    Read::read_to_end(&mut file, &mut buffer)?;
    Ok(buffer
        .chunks_exact(4)
        .take(expected_samples as usize)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Load WAV audio using hound
fn load_wav_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader =
        hound::WavReader::open(path).with_context(|| format!("Failed to open WAV: {:?}", path))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|ch| (ch[0] + ch.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

/// Load audio using symphonia (FLAC, MP3, AAC, OGG, etc.)
#[cfg(feature = "symphonia")]
fn load_symphonia_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{SampleBuffer, SignalSpec};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // Open file
    let file = fs::File::open(path).with_context(|| format!("Failed to open: {:?}", path))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create hint from extension
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe format
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .with_context(|| "Failed to probe audio format")?;

    let mut format = probed.format;

    // Find first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("No audio track found")?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let n_channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    // Create decoder
    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("Failed to create decoder")?;

    // Decode all packets
    let mut samples = Vec::new();
    let mut sample_buf = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        let spec = SignalSpec::new(decoded.spec().rate, decoded.spec().channels);

        if sample_buf.is_none() {
            let duration = decoded.capacity() as u64;
            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
        }

        if let Some(ref mut buf) = sample_buf {
            buf.copy_interleaved_ref(decoded);
            samples.extend_from_slice(buf.samples());
        }
    }

    // Convert to mono if stereo
    let mono = if n_channels == 2 {
        samples
            .chunks(2)
            .map(|ch| (ch[0] + ch.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

// ============================================================================
// 105D Feature Extraction (Same as Training)
// ============================================================================

fn compute_spectral_flux_std(spectrum: &[f32]) -> f32 {
    if spectrum.len() < 2 {
        return 0.0;
    }
    let flux: Vec<f32> = spectrum.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let mean = flux.iter().sum::<f32>() / flux.len() as f32;
    let variance = flux.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flux.len() as f32;
    variance.sqrt()
}

// ============================================================================
// Spectrum and FFT
// ============================================================================

fn compute_spectrum(audio: &[f32], n_fft: usize) -> Vec<f32> {
    let n_fft = n_fft.max(64).min(4096);
    let mut real = vec![0.0f32; n_fft];
    let mut imag = vec![0.0f32; n_fft];
    let start = audio.len().saturating_sub(n_fft) / 2;
    for (i, &s) in audio.iter().skip(start).take(n_fft).enumerate() {
        real[i] = s;
    }
    fft_inplace(&mut real, &mut imag);
    (0..=n_fft / 2)
        .map(|k| (real[k] * real[k] + imag[k] * imag[k]).sqrt())
        .collect()
}

fn fft_inplace(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    if n <= 1 {
        return;
    }
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle_step = -std::f32::consts::PI / half_len as f32;
        for i in (0..n).step_by(len) {
            for j in 0..half_len {
                let angle = angle_step * j as f32;
                let (tw_r, tw_i) = (angle.cos(), angle.sin());
                let (even_idx, odd_idx) = (i + j, i + j + half_len);
                let (t_r, t_i) = (
                    real[odd_idx] * tw_r - imag[odd_idx] * tw_i,
                    real[odd_idx] * tw_i + imag[odd_idx] * tw_r,
                );
                real[odd_idx] = real[even_idx] - t_r;
                imag[odd_idx] = imag[even_idx] - t_i;
                real[even_idx] = real[even_idx] + t_r;
                imag[even_idx] = imag[even_idx] + t_i;
            }
        }
        len *= 2;
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }
    let manifest_path = PathBuf::from(&args[1]);

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║       Pre-Extract 105D Features for BEANS-Zero Evaluation            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    let base_path = manifest_path.parent().unwrap_or(Path::new("."));

    // Load manifest
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    println!("\nLoading manifest from: {:?}", manifest_path);
    println!(
        "Dataset: {} ({} samples)",
        manifest.dataset, manifest.n_samples
    );

    // Check if cache already exists
    let cache_path = base_path.join("feature_cache_eval/all_features.bin");
    if cache_path.exists() {
        println!("\nCache already exists at {:?}", cache_path);
        println!("  Delete it to re-extract features.");
        return Ok(());
    }

    // Create cache directory
    let cache_dir = cache_path.parent().unwrap();
    fs::create_dir_all(cache_dir)?;

    println!("\nExtracting 105D features from ALL samples...");

    let processed = AtomicUsize::new(0);
    let start = Instant::now();

    // Extract features in parallel
    let all_features: Vec<CachedFeatures> = manifest
        .samples
        .par_iter()
        .filter_map(|s| {
            // Load audio (auto-detect format)
            let audio_result = load_audio_file(&base_path.join(&s.audio_file), Some(s.n_samples));

            let (audio, sr) = match audio_result {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("  Error loading {}: {}", s.audio_file, e);
                    return None;
                }
            };

            // Extract features
            let extractor = MicroDynamicsExtractor::new(sr);
            let base_features = match extractor.extract(&audio) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("  Feature extraction failed for {}: {}", s.audio_file, e);
                    return None;
                }
            };
            let spectrum = compute_spectrum(&audio, 2048);

            // Create 105D feature vector
            let mut features = vec![0.0f32; 105];

            // Copy base features (from MicroDynamicsFeatures)
            features[0] = base_features.attack_time_ms;
            features[1] = base_features.decay_time_ms;
            features[2] = base_features.sustain_level;
            features[3] = base_features.vibrato_rate_hz;
            features[4] = base_features.vibrato_depth;
            features[5] = base_features.jitter;
            features[6] = base_features.shimmer;
            features[7] = base_features.harmonicity;
            features[8] = base_features.spectral_flatness;
            features[9] = base_features.harmonic_to_noise_ratio;
            features[10] = base_features.spectral_flux;
            features[11] = base_features.median_ici_ms;
            features[12] = base_features.onset_rate_hz;
            features[13] = base_features.ici_coefficient_of_variation;

            // Add MFCCs (13)
            for (i, &mfcc) in base_features.mfcc.iter().enumerate() {
                features[14 + i] = mfcc;
            }

            // Add duration
            features[27] = audio.len() as f32 / sr as f32 * 1000.0; // duration_ms

            // Add spectrum-derived features
            features[28] = compute_spectral_flux_std(&spectrum);

            // Fill remaining with spectrum bands
            for (i, &s) in spectrum.iter().take(76).enumerate() {
                features[29 + i] = s;
            }

            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 5000 == 4999 {
                println!("  Processed {}/{}", count + 1, manifest.n_samples);
            }

            Some(CachedFeatures {
                features,
                audio_file: s.audio_file.clone(),
            })
        })
        .collect();

    let elapsed = start.elapsed();
    println!("Extraction completed in {:.2}s", elapsed.as_secs_f32());
    println!("Cached {} feature vectors", all_features.len());

    // Save to binary file
    println!("\nSaving to {:?}...", cache_path);

    let mut file = BufWriter::new(fs::File::create(&cache_path)?);

    // Write header: [magic, n_samples, feature_dim]
    file.write_all(&0x46454154u32.to_le_bytes())?; // "FEAT"
    file.write_all(&(all_features.len() as u32).to_le_bytes())?;
    file.write_all(&105u32.to_le_bytes())?;

    // Write features
    for cf in &all_features {
        for &val in &cf.features {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    println!("Done!");
    Ok(())
}
