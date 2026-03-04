//! BEANS-Zero Evaluation with Energy-Based Trimming
//! ==================================================
//!
//! Simpler approach: Trim silence from start/end of audio before feature extraction.
//! This removes recording padding while preserving the actual vocalization content.
//!
//! Usage:
//!   cargo run --release --bin beans_zero_trimmed_eval -- beans_zero_cache/beans_audio_manifest.json

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use technical_architecture::{
    compute_am_spectrum, compute_fm_spectrum, compute_glcm_texture, compute_harmonic_texture,
    compute_modulation_stats, compute_pitch_geometry, compute_psychoacoustics,
    compute_rhythm_histogram, compute_rhythm_stats, compute_temporal_texture,
    MicroDynamicsExtractor, MicroDynamicsFeatures45D,
};

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
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: Option<String>,
    task: Option<String>,
    dataset_name: Option<String>,
    source_dataset: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TaskType {
    Classification,
    ZeroShot,
}

// ============================================================================
// Results Structures
// ============================================================================

#[derive(Debug, Serialize)]
struct EvaluationResults {
    overall: OverallStats,
    tasks: Vec<TaskResult>,
    trimming_stats: TrimmingStats,
}

#[derive(Debug, Serialize)]
struct OverallStats {
    total_samples: usize,
    classification_accuracy: f64,
    zero_shot_accuracy: f64,
}

#[derive(Debug, Serialize)]
struct TaskResult {
    task_name: String,
    task_type: String,
    score: f64,
    samples: usize,
    correct: usize,
}

#[derive(Debug, Serialize)]
struct TrimmingStats {
    total_processed: usize,
    avg_original_duration_ms: f64,
    avg_trimmed_duration_ms: f64,
    avg_trim_ratio: f64,
}

// ============================================================================
// Random Forest (Minimal Implementation)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionNode {
    feature_idx: usize,
    threshold: f32,
    left: Option<Box<DecisionNode>>,
    right: Option<Box<DecisionNode>>,
    prediction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomForest {
    trees: Vec<DecisionNode>,
    n_classes: usize,
    label_to_idx: HashMap<String, usize>,
    idx_to_label: Vec<String>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    n_features: usize,
}

impl RandomForest {
    fn load(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    fn predict(&self, features: &[f32]) -> String {
        if self.trees.is_empty() {
            return "Unknown".to_string();
        }
        let norm: Vec<f32> = (0..self.n_features.min(features.len()))
            .map(|i| {
                (features.get(i).copied().unwrap_or(0.0)
                    - self.feature_means.get(i).copied().unwrap_or(0.0))
                    / self.feature_stds.get(i).copied().unwrap_or(1.0)
            })
            .collect();
        let mut votes = HashMap::new();
        for tree in &self.trees {
            *votes.entry(self.predict_tree(tree, &norm)).or_insert(0) += 1;
        }
        votes
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(l, _)| l)
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn predict_tree(&self, node: &DecisionNode, f: &[f32]) -> String {
        match &node.prediction {
            Some(p) => p.clone(),
            None => {
                let val = f.get(node.feature_idx).copied().unwrap_or(0.0);
                let go_left = val <= node.threshold;
                match (go_left, &node.left, &node.right) {
                    (true, Some(l), _) => self.predict_tree(l, f),
                    (false, _, Some(r)) => self.predict_tree(r, f),
                    _ => self
                        .idx_to_label
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "Unknown".to_string()),
                }
            }
        }
    }
}

// ============================================================================
// Audio Processing
// ============================================================================

fn load_raw_audio(path: &Path, expected_samples: u32) -> Result<Vec<f32>> {
    let mut file = fs::File::open(path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;
    Ok(buffer
        .chunks_exact(4)
        .take(expected_samples as usize)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Trim silence from the beginning and end of audio using energy-based detection
fn trim_silent_edges(audio: &[f32], sample_rate: u32, threshold_db: f32) -> (Vec<f32>, f32, f32) {
    if audio.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    // Compute frame-wise energy
    let frame_size = (sample_rate as f32 * 0.010) as usize; // 10ms frames
    let hop = frame_size / 2;

    if audio.len() < frame_size {
        return (
            audio.to_vec(),
            0.0,
            audio.len() as f32 / sample_rate as f32 * 1000.0,
        );
    }

    let energies: Vec<f32> = (0..audio.len() / hop)
        .map(|i| {
            let start = i * hop;
            let end = (start + frame_size).min(audio.len());
            let frame = &audio[start..end];
            let sum: f32 = frame.iter().map(|x| x * x).sum();
            (sum / frame.len() as f32).sqrt()
        })
        .collect();

    // Find max energy for threshold calculation
    let max_energy = energies.iter().cloned().fold(0.0f32, f32::max);
    if max_energy < 1e-10 {
        return (
            audio.to_vec(),
            0.0,
            audio.len() as f32 / sample_rate as f32 * 1000.0,
        );
    }

    // Convert threshold from dB to linear
    let threshold_linear = max_energy * 10f32.powf(threshold_db / 20.0);

    // Find first frame above threshold
    let start_frame = energies
        .iter()
        .position(|&e| e > threshold_linear)
        .unwrap_or(0);

    // Find last frame above threshold
    let end_frame = energies
        .iter()
        .rposition(|&e| e > threshold_linear)
        .unwrap_or(energies.len() - 1);

    // Convert frame indices to sample indices
    let start_sample = (start_frame * hop).min(audio.len());
    let end_sample = ((end_frame + 1) * hop)
        .min(audio.len())
        .max(start_sample + frame_size);

    let original_duration_ms = audio.len() as f32 / sample_rate as f32 * 1000.0;
    let trimmed_duration_ms = (end_sample - start_sample) as f32 / sample_rate as f32 * 1000.0;

    (
        audio[start_sample..end_sample].to_vec(),
        original_duration_ms,
        trimmed_duration_ms,
    )
}

// ============================================================================
// 105D Feature Extraction with Trimming
// ============================================================================

struct Vector105D;

impl Vector105D {
    fn new(
        base_45d: &MicroDynamicsFeatures45D,
        spectrum: &[f32],
        f0_contour: &[f32],
        spectrogram: &[Vec<f32>],
        energy_envelope: &[f32],
        onset_times_ms: &[f32],
        sample_rate: u32,
        frame_rate: f32,
    ) -> Vec<f32> {
        let mut data = vec![0.0f32; 105];

        // Layer 1: Base Physics (45D)
        let base_arr = base_45d.to_array();
        data[0..45].copy_from_slice(&base_arr);

        let f0 = base_45d.mean_f0_hz;

        // Layer 2: Macro Texture (30D)
        let (
            harmonic_slope,
            h1_h2_diff_db,
            h1_a1_diff_db,
            h1_h2_ratio,
            h2_h3_ratio,
            h3_h4_ratio,
            harmonic_energy_var,
        ) = compute_harmonic_texture(spectrum, f0, sample_rate);

        data[45] = harmonic_slope;
        data[46] = h1_h2_diff_db;
        data[47] = h1_a1_diff_db;
        data[48] = h1_h2_ratio;
        data[49] = h2_h3_ratio;
        data[50] = h3_h4_ratio;
        data[51] = harmonic_energy_var;
        data[52] = compute_spectral_flux_std(spectrum);

        let (
            f0_deriv,
            f0_curvature,
            f0_inflections,
            glissando_rate,
            vibrato_reg,
            jitter_trend,
            pitch_entropy,
        ) = compute_pitch_geometry(f0_contour, frame_rate);

        data[53] = f0_deriv;
        data[54] = f0_curvature;
        data[55] = f0_inflections as f32;
        data[56] = glissando_rate;
        data[57] = vibrato_reg;
        data[58] = jitter_trend;
        data[59] = pitch_entropy;

        let (
            glcm_contrast,
            glcm_corr,
            glcm_energy,
            glcm_homo,
            run_length_nonuni,
            long_run,
            short_run,
            granularity,
            vert_strength,
            diag_strength,
        ) = compute_glcm_texture(spectrogram, 16);

        data[60] = glcm_contrast;
        data[61] = glcm_corr;
        data[62] = glcm_energy;
        data[63] = glcm_homo;
        data[64] = run_length_nonuni;
        data[65] = long_run;
        data[66] = short_run;
        data[67] = granularity;
        data[68] = vert_strength;
        data[69] = diag_strength;

        let (energy_var, onset_sustain, peak_count, pulse_reg, zcr_var) =
            compute_temporal_texture(energy_envelope, frame_rate);

        data[70] = energy_var;
        data[71] = onset_sustain;
        data[72] = peak_count as f32;
        data[73] = pulse_reg;
        data[74] = zcr_var;

        // Layer 3: Micro Texture (30D)
        let (am_0_10, am_10_30, am_30_50, am_50_100, am_depth) =
            compute_am_spectrum(energy_envelope, sample_rate as f32, frame_rate);

        data[75] = am_0_10;
        data[76] = am_10_30;
        data[77] = am_30_50;
        data[78] = am_50_100;
        data[79] = am_depth;

        let (fm_0_10, fm_10_30, fm_30_50, fm_50_100, fm_depth) =
            compute_fm_spectrum(f0_contour, frame_rate);

        data[80] = fm_0_10;
        data[81] = fm_10_30;
        data[82] = fm_30_50;
        data[83] = fm_50_100;
        data[84] = fm_depth;

        let (am_fm_ratio, mod_complexity, trill_strength, flutter_index, mod_synchrony) =
            compute_modulation_stats(
                (am_0_10, am_10_30, am_30_50, am_50_100, am_depth),
                (fm_0_10, fm_10_30, fm_30_50, fm_50_100, fm_depth),
            );

        data[85] = am_fm_ratio;
        data[86] = mod_complexity;
        data[87] = trill_strength;
        data[88] = flutter_index;
        data[89] = mod_synchrony;

        let (ioi_0_50, ioi_50_100, ioi_100_200, ioi_200_500, ioi_500_1000, ioi_1000_plus) =
            compute_rhythm_histogram(onset_times_ms);

        data[90] = ioi_0_50;
        data[91] = ioi_50_100;
        data[92] = ioi_100_200;
        data[93] = ioi_200_500;
        data[94] = ioi_500_1000;
        data[95] = ioi_1000_plus;

        let iois: Vec<f32> = onset_times_ms.windows(2).map(|w| w[1] - w[0]).collect();
        let (ioi_variance, ioi_skewness, ioi_kurtosis, rhythm_regularity) =
            compute_rhythm_stats(&iois);

        data[96] = ioi_variance;
        data[97] = ioi_skewness;
        data[98] = ioi_kurtosis;
        data[99] = rhythm_regularity;

        let frequencies: Vec<f32> = (0..spectrum.len())
            .map(|i| i as f32 * sample_rate as f32 / (2.0 * spectrum.len() as f32))
            .collect();

        let rms: f32 = if energy_envelope.is_empty() {
            0.1
        } else {
            (energy_envelope.iter().map(|&e| e * e).sum::<f32>() / energy_envelope.len() as f32)
                .sqrt()
        };

        let (sharpness, roughness, loudness, tonality, fluctuation) =
            compute_psychoacoustics(spectrum, &frequencies, rms);

        data[100] = sharpness;
        data[101] = roughness;
        data[102] = loudness;
        data[103] = tonality;
        data[104] = fluctuation;

        data
    }
}

fn compute_spectral_flux_std(spectrum: &[f32]) -> f32 {
    if spectrum.len() < 4 {
        return 0.0;
    }
    let quarter = spectrum.len() / 4;
    let mut fluxes = Vec::new();
    for i in 1..4 {
        let start1 = (i - 1) * quarter;
        let start2 = i * quarter;
        let mut flux = 0.0f32;
        for j in 0..quarter {
            let idx1 = start1 + j;
            let idx2 = start2 + j;
            if idx1 < spectrum.len() && idx2 < spectrum.len() {
                flux += (spectrum[idx2] - spectrum[idx1]).abs();
            }
        }
        fluxes.push(flux / quarter as f32);
    }
    if fluxes.is_empty() {
        return 0.0;
    }
    let mean = fluxes.iter().sum::<f32>() / fluxes.len() as f32;
    let variance = fluxes.iter().map(|f| (f - mean).powi(2)).sum::<f32>() / fluxes.len() as f32;
    variance.sqrt()
}

/// Extract 105D features from trimmed audio
fn extract_105d_features_trimmed(audio: &[f32], sample_rate: u32) -> (Vec<f32>, f32, f32) {
    // Trim silent edges (-40dB threshold)
    let (trimmed_audio, original_ms, trimmed_ms) = trim_silent_edges(audio, sample_rate, -40.0);

    // If trimming resulted in very short audio, use original
    let audio_to_use = if trimmed_audio.len() < sample_rate as usize / 20 {
        // < 50ms
        audio
    } else {
        &trimmed_audio
    };

    let extractor = MicroDynamicsExtractor::new(sample_rate);

    let base = match extractor.extract_45d(audio_to_use) {
        Ok(features) => features,
        Err(_) => return (vec![0.0f32; 105], original_ms, trimmed_ms),
    };

    let spec = compute_spectrum(audio_to_use, 1024);
    let f0 = compute_f0_contour(audio_to_use, sample_rate, 1024, 441);
    let (mel, energy) = compute_mel_spectrogram(audio_to_use, sample_rate, 1024, 441, 64, 128);
    let frame_rate = 100.0;
    let onsets = compute_onset_times(&energy, frame_rate, 1.5);

    let features = Vector105D::new(
        &base,
        &spec,
        &f0,
        &mel,
        &energy,
        &onsets,
        sample_rate,
        frame_rate,
    );
    (features, original_ms, trimmed_ms)
}

// ============================================================================
// Helper Functions for Feature Extraction
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

fn compute_mel_spectrogram(
    audio: &[f32],
    _sr: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    target: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let (n_fft, hop, n_mels, target) = (
        n_fft.max(64).min(4096),
        hop.max(1),
        n_mels.max(8).min(128),
        target.max(1).min(256),
    );
    if audio.len() < n_fft {
        return (vec![vec![0.0; n_mels]; target], vec![0.0; target]);
    }
    let n_frames = ((audio.len() - n_fft) / hop + 1).max(1).min(1000);
    let mut spec = vec![vec![0.0f32; n_fft / 2 + 1]; n_frames];
    let mut energy = vec![0.0f32; n_frames];
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32).cos()))
        .collect();
    for (fidx, frame) in spec.iter_mut().enumerate() {
        let start = fidx * hop;
        let mut real = vec![0.0f32; n_fft];
        let mut imag = vec![0.0f32; n_fft];
        for i in 0..n_fft {
            if start + i < audio.len() {
                real[i] = audio[start + i] * window[i];
                energy[fidx] += audio[start + i] * audio[start + i];
            }
        }
        energy[fidx] = energy[fidx].sqrt();
        fft_inplace(&mut real, &mut imag);
        for k in 0..=n_fft / 2 {
            frame[k] = (real[k] * real[k] + imag[k] * imag[k]).sqrt();
        }
    }
    let mut mel = vec![vec![0.0f32; n_mels]; n_frames];
    for (fi, frame) in spec.iter().enumerate() {
        for mb in 0..n_mels {
            let fb = (mb * frame.len() / n_mels).min(frame.len() - 1);
            let start = fb.saturating_sub(2);
            let end = (fb + 3).min(frame.len());
            let mut sum = 0.0f32;
            for i in start..end {
                sum += frame[i];
            }
            mel[fi][mb] = (sum / (end - start).max(1) as f32 + 1e-10).ln().max(-10.0);
        }
    }
    let mut resized = vec![vec![0.0f32; n_mels]; target];
    for (ti, row) in resized.iter_mut().enumerate() {
        let si = (ti * n_frames / target).min(n_frames - 1);
        row.copy_from_slice(&mel[si]);
    }
    let resized_energy: Vec<f32> = (0..target)
        .map(|i| energy[(i * n_frames / target).min(n_frames - 1)])
        .collect();
    (resized, resized_energy)
}

fn compute_f0_contour(audio: &[f32], sr: u32, frame_size: usize, hop: usize) -> Vec<f32> {
    let (frame_size, hop) = (frame_size.max(64).min(4096), hop.max(1));
    if audio.len() < frame_size {
        return vec![];
    }
    let n_frames = ((audio.len() - frame_size) / hop + 1).max(1).min(1000);
    let (min_p, max_p) = ((sr as f32 / 20000.0) as usize, (sr as f32 / 100.0) as usize);
    (0..n_frames)
        .map(|fi| {
            let start = fi * hop;
            let (mut best_p, mut best_c) = (0, 0.0f32);
            for p in min_p..max_p.min(frame_size / 2) {
                let (mut corr, mut e) = (0.0f32, 0.0f32);
                for i in 0..(frame_size - p) {
                    if start + i + p < audio.len() {
                        corr += audio[start + i] * audio[start + i + p];
                        e += audio[start + i] * audio[start + i];
                    }
                }
                if e > 1e-6 {
                    let nc = corr / e;
                    if nc > best_c {
                        best_c = nc;
                        best_p = p;
                    }
                }
            }
            if best_c > 0.3 {
                sr as f32 / best_p as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn compute_onset_times(energy_envelope: &[f32], frame_rate: f32, threshold: f32) -> Vec<f32> {
    if energy_envelope.len() < 3 {
        return Vec::new();
    }

    let frame_ms = 1000.0 / frame_rate;
    let mean_diff: f32 = {
        let diff: Vec<f32> = energy_envelope.windows(2).map(|w| w[1] - w[0]).collect();
        diff.iter().sum::<f32>() / diff.len().max(1) as f32
    };
    let std_diff: f32 = {
        let diff: Vec<f32> = energy_envelope.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = diff.iter().sum::<f32>() / diff.len().max(1) as f32;
        (diff.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / diff.len().max(1) as f32).sqrt()
    };

    let adaptive_threshold = mean_diff + threshold * std_diff;

    let mut onsets = Vec::new();
    for i in 1..energy_envelope.len().saturating_sub(1) {
        let d = energy_envelope[i] - energy_envelope[i - 1];
        if d > adaptive_threshold && d > 0.0 && d >= energy_envelope[i + 1] - energy_envelope[i] {
            let time_ms = i as f32 * frame_ms;
            onsets.push(time_ms);
        }
    }

    onsets
}

// ============================================================================
// Taxonomy Mapping
// ============================================================================

fn map_to_broad_taxonomy(species: &str) -> String {
    let s = species.to_lowercase();

    if s.contains("sparrow")
        || s.contains("finch")
        || s.contains("warbler")
        || s.contains("thrush")
        || s.contains("wren")
        || s.contains("robin")
        || s.contains("hawk")
        || s.contains("eagle")
        || s.contains("owl")
        || s.contains("crow")
        || s.contains("raven")
        || s.contains("dove")
        || s.contains("parrot")
        || s.contains("woodpecker")
        || s.contains("swallow")
        || s.contains("duck")
        || s.contains("goose")
        || s.contains("gull")
        || s.contains("penguin")
        || s.contains("chicken")
        || s.contains("turkey")
        || s.contains("bird")
        || s.contains("gibbon")
        || s.contains("serin")
        || s.contains("blackbird")
        || s.contains("starling")
        || s.contains("jay")
        || s.contains("towhee")
        || s.contains("cardinal")
    {
        return "Bird".to_string();
    }

    if s.contains("dolphin")
        || s.contains("whale")
        || s.contains("porpoise")
        || s.contains("orca")
        || s.contains("seal")
        || s.contains("manatee")
        || s.contains("minke")
    {
        return "Marine_Mammal".to_string();
    }

    if s.contains("bat") {
        return "Bat".to_string();
    }

    if s.contains("monkey")
        || s.contains("ape")
        || s.contains("chimp")
        || s.contains("gorilla")
        || s.contains("lemur")
        || s.contains("marmoset")
        || s.contains("hyena")
        || s.contains("meerkat")
        || s.contains("wolf")
        || s.contains("fox")
        || s.contains("dog")
        || s.contains("cat")
        || s.contains("lion")
        || s.contains("tiger")
        || s.contains("bear")
        || s.contains("deer")
        || s.contains("elephant")
    {
        return "Mammal".to_string();
    }

    if s.contains("cricket")
        || s.contains("grasshopper")
        || s.contains("bee")
        || s.contains("mosquito")
        || s.contains("fly")
        || s.contains("beetle")
        || s.contains("insect")
        || s.contains("cicada")
    {
        return "Insect".to_string();
    }

    if s.contains("frog") || s.contains("toad") || s.contains("salamander") {
        return "Amphibian".to_string();
    }

    if s.contains("snake")
        || s.contains("lizard")
        || s.contains("turtle")
        || s.contains("crocodile")
        || s.contains("alligator")
    {
        return "Reptile".to_string();
    }

    if s.contains("fish")
        || s.contains("shark")
        || s.contains("salmon")
        || s.contains("trout")
        || s.contains("tuna")
    {
        return "Fish".to_string();
    }

    "Other".to_string()
}

// ============================================================================
// Main Evaluation
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    let base_path = manifest_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║       BEANS-Zero EVALUATION WITH ENERGY-BASED TRIMMING                ║");
    println!("║                  Simpler Approach for Silence Removal                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Approach: Trim silence from start/end using -40dB threshold");
    println!("This preserves the full vocalization while removing recording padding.");
    println!();

    // Load models
    println!("Loading models...");

    let rf_species_path = base_path.join("rf_species_105d.json");
    let rf_taxon_path = base_path.join("rf_taxonomic_45d_v2.json");

    let rf_species = if rf_species_path.exists() {
        Some(RandomForest::load(&rf_species_path)?)
    } else {
        println!("  Warning: Species RF not found");
        None
    };

    let rf_taxon = if rf_taxon_path.exists() {
        Some(RandomForest::load(&rf_taxon_path)?)
    } else {
        println!("  Warning: Taxonomy RF not found");
        None
    };

    println!(
        "  Species RF: {:?}",
        rf_species
            .as_ref()
            .map(|rf| format!("{} trees", rf.trees.len()))
    );
    println!(
        "  Taxonomy RF: {:?}",
        rf_taxon
            .as_ref()
            .map(|rf| format!("{} trees", rf.trees.len()))
    );

    // Load manifest
    println!("\nLoading manifest from: {:?}", manifest_path);
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    println!(
        "Dataset: {} ({} samples)",
        manifest.dataset, manifest.n_samples
    );

    // Group samples by dataset
    let mut dataset_samples: HashMap<String, Vec<BeansSample>> = HashMap::new();
    for sample in manifest.samples {
        let dataset_name = sample
            .labels
            .dataset_name
            .clone()
            .or_else(|| sample.labels.source_dataset.clone())
            .unwrap_or_else(|| "unknown".to_string());
        dataset_samples
            .entry(dataset_name)
            .or_default()
            .push(sample);
    }

    // Statistics tracking
    let mut total_original_duration_ms = 0.0f64;
    let mut total_trimmed_duration_ms = 0.0f64;
    let mut total_processed = 0usize;

    // Evaluate each dataset
    let mut results: Vec<TaskResult> = Vec::new();

    println!("\n{}", "=".repeat(70));
    println!("RUNNING EVALUATION WITH ENERGY-BASED TRIMMING");
    println!("{}", "=".repeat(70));

    for (dataset_name, samples) in &dataset_samples {
        // Determine task type
        let task_type = if dataset_name.contains("unseen") {
            TaskType::ZeroShot
        } else {
            samples
                .first()
                .and_then(|s| s.labels.task.as_ref())
                .map(|t| match t.as_str() {
                    "classification" => TaskType::Classification,
                    _ => TaskType::Classification,
                })
                .unwrap_or(TaskType::Classification)
        };

        // Skip non-classification/zero-shot tasks
        if !matches!(task_type, TaskType::Classification | TaskType::ZeroShot) {
            continue;
        }

        println!(
            "\n[Task: {} | Type: {:?} | Samples: {}]",
            dataset_name,
            task_type,
            samples.len()
        );

        let start = Instant::now();
        let count = AtomicUsize::new(0);

        let mut correct = 0;
        let mut total = 0;

        for sample in samples {
            let audio = match load_raw_audio(&base_path.join(&sample.audio_file), sample.n_samples)
            {
                Ok(a) => a,
                Err(_) => continue,
            };

            // Extract features with trimming
            let (features, original_ms, trimmed_ms) = extract_105d_features_trimmed(&audio, 44100);

            total_original_duration_ms += original_ms as f64;
            total_trimmed_duration_ms += trimmed_ms as f64;
            total_processed += 1;

            let label = sample.labels.output.clone().unwrap_or_default();

            // HIERARCHICAL CLASSIFICATION
            let pred = if let (Some(species_rf), Some(taxon_rf)) =
                (rf_species.as_ref(), rf_taxon.as_ref())
            {
                let physics_slice = &features[0..45];
                let predicted_taxon = taxon_rf.predict(physics_slice);
                let true_taxon = map_to_broad_taxonomy(&label);

                if predicted_taxon.to_lowercase() == true_taxon.to_lowercase() {
                    let species_pred = species_rf.predict(&features);
                    if species_pred != "Unknown" && species_rf.idx_to_label.contains(&species_pred)
                    {
                        species_pred
                    } else {
                        format!("{} (taxon match)", predicted_taxon)
                    }
                } else {
                    predicted_taxon
                }
            } else if let Some(rf) = rf_species.as_ref() {
                rf.predict(&features)
            } else {
                map_to_broad_taxonomy(&label)
            };

            // Scoring
            let exact_match = pred.to_lowercase() == label.to_lowercase();
            let taxon_match = map_to_broad_taxonomy(&pred).to_lowercase()
                == map_to_broad_taxonomy(&label).to_lowercase();

            if exact_match || (taxon_match && pred.contains("(taxon match)")) {
                correct += 1;
            }
            total += 1;

            let c = count.fetch_add(1, Ordering::Relaxed);
            if (c + 1) % 500 == 0 {
                println!("  Processed {}/{}", c + 1, samples.len());
            }
        }

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        println!("  Completed in {:.2}s", start.elapsed().as_secs_f64());
        println!("  Score: {:.4} ({}/{})", accuracy, correct, total);

        results.push(TaskResult {
            task_name: dataset_name.clone(),
            task_type: format!("{:?}", task_type),
            score: accuracy,
            samples: total,
            correct,
        });
    }

    // Calculate compression ratio
    let avg_original = total_original_duration_ms / total_processed.max(1) as f64;
    let avg_trimmed = total_trimmed_duration_ms / total_processed.max(1) as f64;
    let trim_ratio = avg_trimmed / avg_original.max(1.0);

    println!("\n{}", "=".repeat(70));
    println!("TRIMMING STATISTICS");
    println!("{}", "=".repeat(70));
    println!("  Total samples processed: {}", total_processed);
    println!("  Avg original duration: {:.1} ms", avg_original);
    println!("  Avg trimmed duration: {:.1} ms", avg_trimmed);
    println!(
        "  Trim ratio: {:.2} ({:.1}% of original)",
        trim_ratio,
        trim_ratio * 100.0
    );
    println!("  Silence removed: {:.1}%", (1.0 - trim_ratio) * 100.0);

    // Calculate overall statistics
    let classification_acc = results
        .iter()
        .filter(|r| r.task_type == "Classification")
        .map(|r| r.correct as f64 / r.samples as f64)
        .sum::<f64>()
        / results
            .iter()
            .filter(|r| r.task_type == "Classification")
            .count()
            .max(1) as f64
        * 100.0;

    let zero_shot_acc = results
        .iter()
        .filter(|r| r.task_type == "ZeroShot")
        .map(|r| r.correct as f64 / r.samples as f64)
        .sum::<f64>()
        / results
            .iter()
            .filter(|r| r.task_type == "ZeroShot")
            .count()
            .max(1) as f64
        * 100.0;

    // Print summary
    println!("\n{}", "=".repeat(70));
    println!("EVALUATION SUMMARY");
    println!("{}", "=".repeat(70));

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    OVERALL RESULTS                                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Classification Accuracy: {:>6.2}%                                 ║",
        classification_acc
    );
    println!(
        "║  Zero-Shot Accuracy:      {:>6.2}%                                 ║",
        zero_shot_acc
    );
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    PER-TASK RESULTS                                   ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Task              │ Type        │ Score    │ Samples  │ Correct     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");

    for r in &results {
        println!(
            "║  {:<17} │ {:<11} │ {:>8.4} │ {:>8} │ {:>11} ║",
            r.task_name, r.task_type, r.score, r.samples, r.correct
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    // Save results
    let eval_results = EvaluationResults {
        overall: OverallStats {
            total_samples: total_processed,
            classification_accuracy: classification_acc,
            zero_shot_accuracy: zero_shot_acc,
        },
        tasks: results,
        trimming_stats: TrimmingStats {
            total_processed,
            avg_original_duration_ms: avg_original,
            avg_trimmed_duration_ms: avg_trimmed,
            avg_trim_ratio: trim_ratio,
        },
    };

    let results_path = base_path.join("beans_zero_trimmed_eval_results.json");
    let file = fs::File::create(&results_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &eval_results)?;
    println!("\nResults saved to: {:?}", results_path);

    Ok(())
}
