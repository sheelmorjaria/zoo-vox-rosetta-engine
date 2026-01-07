#!/usr/bin/env python3
"""
Extract Real Micro-Dynamics from Audio Library
==============================================

This script extracts real micro-dynamics features from the audio
library segments, enabling proper parameter-based synthesis.

Features Extracted:
1. Attack time (ms) - time to reach 90% of peak amplitude
2. Decay time (ms) - time to fall to 10% of peak amplitude
3. Vibrato rate (Hz) - frequency of amplitude modulation
4. Vibrato depth (cents) - extent of pitch modulation
5. Jitter - micro-perturbations in phase
6. Spectral flatness - noise-like quality
7. HNR - harmonic-to-noise ratio

Output:
- Micro-dynamics database with real measurements
- JSON file for quick lookup by phrase key
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
AUDIO_INDEX_MARMOSET = "/home/sheel/birdsong_analysis/src/audio_library/audio_index.json"
AUDIO_INDEX_BAT = "/home/sheel/birdsong_analysis/src/audio_library/bat_audio_index.json"
OUTPUT_DIR = "/home/sheel/birdsong_analysis/src/validation_results"
OUTPUT_PATH = "/home/sheel/birdsong_analysis/src/micro_dynamics_database.json"


def extract_attack_time(audio: np.ndarray, sr: int) -> float:
    """Extract attack time - time to reach 90% of peak amplitude."""
    envelope = np.abs(audio)
    max_env = np.max(envelope)
    threshold = 0.9 * max_env

    # Find first sample above threshold
    above_threshold = np.where(envelope > threshold)[0]
    if len(above_threshold) > 0:
        attack_samples = above_threshold[0]
        return attack_samples / sr * 1000  # Convert to ms
    return 0.0


def extract_decay_time(audio: np.ndarray, sr: int) -> float:
    """Extract decay time - time to fall to 10% of peak amplitude."""
    envelope = np.abs(audio)
    max_env = np.max(envelope)
    threshold = 0.1 * max_env

    # Find peak location
    peak_sample = np.argmax(envelope)

    # Find first sample after peak that falls below threshold
    below_threshold = np.where(envelope[peak_sample:] < threshold)[0]
    if len(below_threshold) > 0:
        decay_samples = below_threshold[0]
        return decay_samples / sr * 1000  # Convert to ms

    # If never falls below threshold, use total duration
    return (len(audio) - peak_sample) / sr * 1000


def extract_vibrato_features(audio: np.ndarray, sr: int) -> tuple:
    """
    Extract vibrato rate and depth from audio.

    Returns:
        (vibrato_rate_hz, vibrato_depth_cents)
    """
    envelope = np.abs(audio)

    # Smooth envelope
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(envelope, sigma=int(sr * 0.002))

    # Find peaks in envelope
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(smoothed, distance=int(sr * 0.05))

    if len(peaks) < 2:
        return 0.0, 0.0

    # Calculate inter-peak intervals
    intervals = np.diff(peaks) / sr

    # Vibrato rate = 1 / mean_interval
    mean_interval = np.mean(intervals)
    if mean_interval > 0:
        vibrato_rate = 1.0 / mean_interval
    else:
        vibrato_rate = 0.0

    # Estimate vibrato depth from peak amplitude variation
    if len(peaks) >= 2:
        peak_amplitudes = smoothed[peaks]
        amplitude_range = np.max(peak_amplitudes) - np.min(peak_amplitudes)
        mean_amplitude = np.mean(peak_amplitudes)

        # Convert to cents (approximate)
        if mean_amplitude > 0:
            depth_ratio = amplitude_range / mean_amplitude
            vibrato_depth_cents = depth_ratio * 50  # Rough approximation
        else:
            vibrato_depth_cents = 0.0
    else:
        vibrato_depth_cents = 0.0

    return vibrato_rate, vibrato_depth_cents


def extract_spectral_flatness(audio: np.ndarray, sr: int) -> float:
    """Extract spectral flatness (ratio of geometric to arithmetic mean)."""
    from scipy.signal import spectrogram

    freqs, times, Sxx = spectrogram(audio, sr)

    # Avoid log(0)
    Sxx_safe = Sxx + 1e-10

    geometric_mean = np.exp(np.mean(np.log(Sxx_safe), axis=0))
    arithmetic_mean = np.mean(Sxx_safe, axis=0)

    flatness = np.mean(geometric_mean / (arithmetic_mean + 1e-10))
    return flatness


def extract_hnr(audio: np.ndarray, sr: int) -> float:
    """Extract Harmonic-to-Noise Ratio (simplified)."""
    # Signal energy
    signal_energy = np.sum(audio**2)

    # Noise estimate (high-frequency component)
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 0.8, btype="high", fs=sr)
    high_freq = filtfilt(b, a, audio)
    noise_energy = np.sum(high_freq**2)

    if noise_energy > 0:
        hnr_linear = signal_energy / noise_energy
        hnr_db = 10 * np.log10(hnr_linear + 1e-10)
        return max(0, hnr_db)  # Clamp to 0 dB minimum

    return 40.0  # Default high HNR if no noise


def extract_jitter(audio: np.ndarray, sr: int) -> float:
    """
    Extract jitter - phase perturbations.

    Simplified: measure zero-crossing rate variability.
    """
    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
    zcr = zero_crossings / len(audio)

    # Normalize to typical range (0-0.1)
    # Higher zcr typically indicates more jitter/noise
    jitter = min(0.1, zcr / 100.0)

    return jitter


def extract_13_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract 13 MFCC coefficients for formant/timbre analysis.

    This expands the acoustic vector space from 4 to 13 MFCCs,
    enabling:
    - Better formant discrimination (vowel quality)
    - Improved timbre morphing (spectral envelope shaping)
    - Enhanced nearest neighbor search in high-dimensional space

    Args:
        audio: Audio samples (numpy array)
        sr: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients to extract (default 13)

    Returns:
        1D numpy array of shape (13,) containing time-averaged MFCCs.
        - MFCC[0]: Energy coefficient (log-energy proxy)
        - MFCC[1-4]: Broad spectral envelope (formants)
        - MFCC[5-13]: Fine spectral structure (harmonics, timbre)

    Note:
        Returns time-averaged MFCCs (mean across frames), not frame-based.
        This is suitable for phrase-level feature representation.
    """
    import warnings
    import librosa

    # Validate input
    if len(audio) == 0:
        return np.zeros(n_mfcc, dtype=np.float32)

    try:
        # Suppress librosa warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Extract MFCCs using librosa
            # n_fft=2048 provides good frequency resolution
            # hop_length=512 provides good time resolution
            mfcc_frame_based = librosa.feature.mfcc(
                y=audio.astype(np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
            )

            # Average across time to get phrase-level features
            # Result shape: (13,)
            mfcc_time_averaged = np.mean(mfcc_frame_based, axis=1)

            return mfcc_time_averaged.astype(np.float32)

    except Exception as e:
        # Fallback: return zeros on error
        return np.zeros(n_mfcc, dtype=np.float32)


def extract_micro_dynamics_from_file(file_path: str, phrase_key: str) -> Dict:
    """Extract all micro-dynamics from a single audio file."""
    try:
        audio, sr = sf.read(file_path)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 22050 if needed
        if sr != 22050:
            from scipy import signal

            num_samples = int(len(audio) * 22050 / sr)
            audio = signal.resample(audio, num_samples)
            sr = 22050

        if len(audio) < sr * 0.01:  # Too short
            return None

        # Extract features
        attack_ms = extract_attack_time(audio, sr)
        decay_ms = extract_decay_time(audio, sr)
        vibrato_rate, vibrato_depth = extract_vibrato_features(audio, sr)
        flatness = extract_spectral_flatness(audio, sr)
        hnr = extract_hnr(audio, sr)
        jitter = extract_jitter(audio, sr)

        # Duration
        duration_ms = len(audio) / sr * 1000

        # Mean F0 from phrase key (parse if available)
        f0_mean = 0.0
        if phrase_key.startswith("F0_"):
            parts = phrase_key.split("_")
            if len(parts) > 1:
                try:
                    f0_mean = float(parts[1])
                except:
                    pass
        elif phrase_key.startswith("FM_"):
            parts = phrase_key.split("_")
            if len(parts) > 1:
                try:
                    f0_mean = float(parts[1]) * 1000  # Convert kHz to Hz
                except:
                    pass

        return {
            "phrase_key": phrase_key,
            "f0_mean": f0_mean,
            "duration_ms": duration_ms,
            "attack_ms": attack_ms,
            "decay_ms": decay_ms,
            "vibrato_rate_hz": vibrato_rate,
            "vibrato_depth_cents": vibrato_depth,
            "jitter": jitter,
            "spectral_flatness": flatness,
            "hnr_db": hnr,
            "sustain_level": 0.7,  # Estimated
        }

    except Exception as e:
        print(f"  ⚠️  Error processing {file_path}: {e}")
        return None


def process_audio_library(audio_index_path: str, species: str) -> Dict:
    """Process all audio files in the library and extract micro-dynamics."""
    print(f"\n{'=' * 80}")
    print(f"EXTRACTING MICRO-DYNAMICS: {species.upper()}")
    print(f"{'=' * 80}")

    with open(audio_index_path, "r") as f:
        audio_index = json.load(f)

    micro_dynamics_db = {}
    processed_count = 0
    total_segments = sum(len(data["segments"]) for data in audio_index["phrases"].values())

    print(
        f"\n🔍 Processing {total_segments} segments from "
        f"{len(audio_index['phrases'])} phrase types..."
    )

    for phrase_key, phrase_data in audio_index["phrases"].items():
        segments = phrase_data["segments"]

        for segment in segments:
            relative_path = segment["relative_path"]
            file_path = Path(audio_index_path).parent / relative_path

            if file_path.exists():
                micro_dynamics = extract_micro_dynamics_from_file(str(file_path), phrase_key)

                if micro_dynamics:
                    micro_dynamics_db[phrase_key] = micro_dynamics
                    processed_count += 1

        if processed_count % 500 == 0:
            print(f"  Processed {processed_count}/{total_segments} segments...")

    print(f"\n✅ Extracted micro-dynamics for {len(micro_dynamics_db)} segments")

    return micro_dynamics_db


def analyze_extracted_features(micro_dynamics_db: Dict):
    """Analyze the extracted micro-dynamics."""
    print(f"\n{'=' * 80}")
    print("MICRO-DYNAMICS ANALYSIS")
    print(f"{'=' * 80}")

    # Collect all feature values
    features = list(micro_dynamics_db.values())

    if not features:
        print("  No features extracted!")
        return

    # Calculate statistics
    feature_names = [
        "attack_ms",
        "decay_ms",
        "vibrato_rate_hz",
        "vibrato_depth_cents",
        "jitter",
        "spectral_flatness",
        "hnr_db",
    ]

    stats = {}
    for feature in feature_names:
        values = [f[feature] for f in features]
        stats[feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

        print(f"\n{feature}:")
        print(f"   Mean: {stats[feature]['mean']:.3f}")
        print(f"   Std:  {stats[feature]['std']:.3f}")
        print(f"   Range: [{stats[feature]['min']:.3f}, {stats[feature]['max']:.3f}]")
        print(f"   Median: {stats[feature]['median']:.3f}")
        print(f"   IQR: [{stats[feature]['q25']:.3f}, {stats[feature]['q75']:.3f}]")

    return stats


def save_micro_dynamics_database(marmoset_db: Dict, bat_db: Dict, stats: Dict):
    """Save the micro-dynamics database to JSON."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    output_data = {
        "extraction_date": str(Path(__file__).stat().st_mtime),
        "species_data": {"marmoset": marmoset_db, "egyptian_bat": bat_db},
        "statistics": stats,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Saved micro-dynamics database to {OUTPUT_PATH}")


def main():
    """Main extraction function."""
    print("=" * 80)
    print("EXTRACTING REAL MICRO-DYNAMICS FROM AUDIO LIBRARY")
    print("=" * 80)

    # Process marmoset
    marmoset_db = process_audio_library(AUDIO_INDEX_MARMOSET, "marmoset")

    # Process bat
    bat_db = process_audio_library(AUDIO_INDEX_BAT, "egyptian_bat")

    # Analyze combined
    print(f"\n{'=' * 80}")
    print("COMBINED ANALYSIS")
    print(f"{'=' * 80}")

    combined_db = {**marmoset_db, **bat_db}
    stats = analyze_extracted_features(combined_db)

    # Save database
    save_micro_dynamics_database(marmoset_db, bat_db, stats)

    print("\n" + "=" * 80)
    print("✅ MICRO-DYNAMICS EXTRACTION COMPLETE!")
    print("=" * 80)

    print(f"\n📂 Output: {OUTPUT_PATH}")
    print(f"   Total segments extracted: {len(combined_db)}")
    print(f"   - Marmoset: {len(marmoset_db)}")
    print(f"   - Bat: {len(bat_db)}")

    print("\n🎯 NEXT STEPS:")
    print("   1. Update t-SNE validation to use real micro-dynamics")
    print("   2. Generate Rust-synthesized audio with real parameters")
    print("   3. Compare congruence scores")
    print("=" * 80)


if __name__ == "__main__":
    main()
