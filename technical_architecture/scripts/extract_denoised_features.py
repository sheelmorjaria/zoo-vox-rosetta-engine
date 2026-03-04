#!/usr/bin/env python3
"""Extract 105D features from denoised BEANS-Zero audio"""

import json
import os
import struct
from pathlib import Path

import numpy as np

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def extract_105d_features(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Simplified 105D feature extraction matching Rust implementation.

    Returns vector of 105 features.
    """
    import librosa
    from scipy.stats import kurtosis, skew

    features = np.zeros(105, dtype=np.float32)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    # Basic stats
    duration_ms = len(audio) / sr * 1000.0
    rms = np.sqrt(np.mean(audio**2))

    # Spectral features using librosa
    try:
        # Compute spectrogram
        n_fft = min(2048, len(audio) // 2)
        hop_length = n_fft // 4

        if n_fft < 64:
            # Audio too short, use simple features
            features[0] = len(audio) / sr * 1000.0  # duration
            features[1] = duration_ms
            features[2] = rms
            return features

        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_mean = S.mean(axis=1)

        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features[0] = cent.mean() / sr  # Normalized

        # Spectral bandwidth
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features[1] = bw.mean() / sr

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features[2] = rolloff.mean() / sr

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
        features[3] = np.clip(flatness.mean(), 0, 1)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        features[4] = contrast.mean()

        # RMS
        rms_feat = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
        features[5] = rms_feat.mean()

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
        features[6] = zcr.mean()

        # MFCCs (first 13)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        for i in range(min(13, mfcc.shape[0])):
            features[7 + i] = mfcc[i].mean()

        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        for i in range(min(12, chroma.shape[0])):
            features[20 + i] = chroma[i].mean()

        # Tonnetz (6)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            for i in range(min(6, tonnetz.shape[0])):
                features[32 + i] = tonnetz[i].mean()
        except Exception:
            pass

        # Spectral flux (delta)
        delta = np.diff(S_mean)
        features[38] = np.mean(np.abs(delta))

        # Energy envelope stats
        envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sr))
        features[39] = envelope.mean()
        features[40] = np.std(envelope)

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features[41] = (
                float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 0.0
            )
        except Exception:
            features[41] = 0.0

        # Duration
        features[42] = duration_ms / 1000.0

        # Statistics
        features[43] = skew(audio)
        features[44] = kurtosis(audio)

        # Fill remaining with spectral band energies
        n_bands = min(60, S.shape[0])
        band_energies = S_mean[:n_bands]
        for i, e in enumerate(band_energies):
            if 45 + i < 105:
                features[45 + i] = e

    except Exception as e:
        # Fallback to basic features
        features[0] = duration_ms
        features[1] = rms

    return features.astype(np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="beans_zero_denoised_test")
    parser.add_argument("--output", default="beans_zero_denoised_test")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load manifest
    manifest_path = input_dir / "beans_audio_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Extracting 105D features from {len(manifest['samples'])} denoised samples...")

    features_dict = {}

    for i, sample in enumerate(manifest["samples"]):
        audio_path = input_dir / sample["audio_file"]
        print(f"  [{i + 1}/{len(manifest['samples'])}] {sample['audio_file']}")

        # Load denoised audio (raw 16-bit PCM)
        audio = np.fromfile(str(audio_path), dtype=np.int16).astype(np.float32) / 32768.0

        # Extract features
        features = extract_105d_features(audio, sr=44100)
        features_dict[sample["id"]] = features

    # Save features in same format as Rust cache
    output_path = output_dir / "feature_cache_eval"
    output_path.mkdir(parents=True, exist_ok=True)

    # Write binary cache
    cache_file = output_path / "all_features.bin"
    with open(cache_file, "wb") as f:
        # Header: magic, n_samples, feature_dim
        f.write(struct.pack("<I", 0x46454154))  # "FEAT"
        f.write(struct.pack("<I", len(features_dict)))
        f.write(struct.pack("<I", 105))

        # Features (sorted by key)
        for sample_id in sorted(features_dict.keys()):
            feat = features_dict[sample_id]
            for v in feat:
                f.write(struct.pack("<f", v))

    print(f"\nSaved {len(features_dict)} feature vectors to {cache_file}")

    # Also save as JSON for debugging
    json_path = output_path / "features.json"
    with open(json_path, "w") as f:
        json.dump({k: v.tolist() for k, v in features_dict.items()}, f, indent=2)
    print(f"Also saved JSON to {json_path}")


if __name__ == "__main__":
    main()
