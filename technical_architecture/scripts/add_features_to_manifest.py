#!/usr/bin/env python3
"""Add 112D features to manifest samples by extracting from audio files."""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def extract_112d_features(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Extract 112D features - copied from build_beans_gallery.py"""
    import librosa
    from scipy.stats import kurtosis, skew

    features = np.zeros(112, dtype=np.float32)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    max_val = np.max(np.abs(audio))
    if max_val > 1e-10:
        audio = audio / max_val

    duration_ms = len(audio) / sr * 1000.0
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0 if len(audio) > 1 else 0.0

    try:
        n_fft = min(2048, len(audio) // 2)
        hop_length = n_fft // 4

        if n_fft < 64:
            features[0] = duration_ms
            features[2] = rms
            return features

        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_mean = S.mean(axis=1)

        features[0] = duration_ms
        features[1] = duration_ms / 1000.0
        features[2] = rms
        features[3] = zcr

        try:
            f0, voiced_flags = librosa.piptrack(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
            f0_vals = f0[voiced_flags > 0] if voiced_flags.any() else np.array([0])
            if len(f0_vals) > 0 and np.any(f0_vals > 0):
                f0_positive = f0_vals[f0_vals > 0]
                if len(f0_positive) > 0:
                    features[4] = np.mean(f0_positive)
                    features[5] = np.std(f0_positive)
                    features[6] = np.min(f0_positive)
                    features[7] = np.max(f0_positive)
        except Exception:
            pass

        cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features[8] = cent.mean()

        bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features[9] = bw.mean()

        rolloff85 = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        )
        rolloff95 = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.95
        )
        features[10] = rolloff85.mean()
        features[11] = rolloff95.mean()
        features[12] = rolloff95.mean() - rolloff85.mean()

        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
        features[13] = np.clip(flatness.mean(), 0, 1)

        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
            features[14] = contrast.mean()
            for i in range(min(6, contrast.shape[0])):
                features[15 + i] = contrast[i].mean()
        except Exception:
            pass

        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            features[21] = np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio)) + 1e-10)
            features[22] = np.mean(np.abs(percussive)) / (np.mean(np.abs(audio)) + 1e-10)
            features[23] = features[21] / (features[22] + 1e-10)
        except Exception:
            pass

        envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sr))
        if len(envelope) > 0:
            peak_idx = np.argmax(envelope)
            peak_val = envelope[peak_idx]
            attack_idx = (
                np.argmax(envelope[: peak_idx + 1] >= 0.9 * peak_val) if peak_idx > 0 else 0
            )
            features[24] = (attack_idx * hop_length / sr) * 1000.0
            if peak_idx < len(envelope) - 1:
                decay_envelope = envelope[peak_idx:]
                decay_idx = np.argmax(decay_envelope < 0.1 * peak_val)
                if decay_idx > 0:
                    features[25] = (decay_idx * hop_length / sr) * 1000.0
            features[26] = peak_val
            features[27] = np.std(envelope)

        try:
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
            )
            for i in range(min(13, mfcc.shape[0])):
                features[28 + i] = mfcc[i].mean()
        except Exception:
            pass

        rms_feat = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
        features[41] = rms_feat.mean()
        features[42] = np.std(rms_feat) if len(rms_feat) > 1 else 0
        features[43] = skew(audio)
        features[44] = kurtosis(audio)
        features[45] = np.percentile(np.abs(audio), 95)

        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
            for i in range(min(12, chroma.shape[0])):
                features[46 + i] = chroma[i].mean()
        except Exception:
            pass

        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            for i in range(min(6, tonnetz.shape[0])):
                features[58 + i] = tonnetz[i].mean()
        except Exception:
            pass

        delta = np.diff(S_mean)
        features[64] = np.mean(np.abs(delta))
        features[65] = np.std(delta) if len(delta) > 1 else 0
        features[66] = np.max(np.abs(delta)) if len(delta) > 0 else 0

        n_bands = 9
        band_size = S.shape[0] // n_bands
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            band_energy = np.mean(S_mean[start:end]) if end <= len(S_mean) else 0
            features[67 + i] = band_energy

        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features[76] = (
                float(tempo) if np.isscalar(tempo) else (float(tempo[0]) if len(tempo) > 0 else 0.0)
            )
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            features[77] = len(onsets) / (duration_ms / 1000.0) if duration_ms > 0 else 0
            if len(onsets) > 1:
                ici = np.diff(onsets) * hop_length / sr * 1000
                features[78] = np.mean(ici)
                features[79] = np.std(ici)
                features[80] = np.median(ici)
        except Exception:
            pass

        try:
            amp_env = librosa.effects.hpss(audio)[0]
            if len(amp_env) > sr // 10:
                amp_fft = np.abs(np.fft.rfft(np.abs(amp_env)))
                freqs = np.fft.rfftfreq(len(amp_env), 1 / sr)
                vib_range = (freqs >= 4) & (freqs <= 8)
                if vib_range.any():
                    vib_peak_idx = np.argmax(amp_fft[vib_range])
                    features[81] = (
                        freqs[vib_range][vib_peak_idx]
                        if vib_peak_idx < len(freqs[vib_range])
                        else 0
                    )
                    features[82] = amp_fft[vib_range].max() / (amp_fft.mean() + 1e-10)
        except Exception:
            pass

        try:
            spec_var = np.std(S, axis=1)
            features[86] = np.mean(spec_var)
            features[87] = np.std(spec_var)
            features[88] = np.max(spec_var)
            spec_skew = np.array([scipy.stats.skew(S[:, i]) for i in range(S.shape[1])])
            spec_kurt = np.array([scipy.stats.kurtosis(S[:, i]) for i in range(S.shape[1])])
            features[89] = np.mean(spec_skew)
            features[90] = np.mean(spec_kurt)
            features[91] = np.std(spec_skew)
            features[92] = np.std(spec_kurt)
        except Exception:
            pass

        features[93] = np.max(audio) - np.min(audio)
        features[94] = np.percentile(audio, 90) - np.percentile(audio, 10)

        try:
            zcr_frames = librosa.feature.zero_crossing_rate(
                audio, frame_length=n_fft, hop_length=hop_length
            )
            features[95] = np.std(zcr_frames)
            features[96] = np.percentile(zcr_frames, 90)
        except Exception:
            pass

        try:
            S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
            entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
            features[97] = np.mean(entropy)
            features[98] = np.std(entropy)
            features[99] = np.min(entropy)
        except Exception:
            pass

        try:
            freqs_spec = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            spec_slope = np.polyfit(freqs_spec[: len(S_mean)], S_mean, 1)
            features[100] = spec_slope[0]
            features[101] = spec_slope[1]
            features[102] = np.mean(np.diff(S_mean) / (freqs_spec[1 : len(S_mean) + 1] + 1e-10))
            features[103] = np.sqrt(
                np.sum(((freqs_spec[: len(S_mean)] - features[8]) ** 2) * S_mean)
                / (S_mean.sum() + 1e-10)
            )
            features[104] = np.sum(np.abs(np.diff(S_mean))) / (S_mean.sum() + 1e-10)
            if mfcc.shape[1] > 1:
                mfcc_delta = np.diff(mfcc, axis=1)
                features[105] = np.mean(np.abs(mfcc_delta))
                features[106] = np.std(mfcc_delta)
            features[107] = features[21] if features[21] > 0 else 0.1
            low_band = S_mean[: len(S_mean) // 4]
            mid_band = S_mean[len(S_mean) // 4 : len(S_mean) // 2]
            high_band = S_mean[len(S_mean) // 2 :]
            features[108] = np.sum(low_band) / (np.sum(S_mean) + 1e-10)
            features[109] = np.sum(mid_band) / (np.sum(S_mean) + 1e-10)
            features[110] = np.sum(high_band) / (np.sum(S_mean) + 1e-10)
            features[111] = features[108] / (features[110] + 1e-10)
        except Exception:
            pass

    except Exception as e:
        features[0] = duration_ms
        features[2] = rms

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add features to manifest")
    parser.add_argument("--manifest", default="beans_zero_manifest_compat.json")
    parser.add_argument("--audio-dir", default="beans_audio_full_rust")
    parser.add_argument("--output", default="beans_zero_manifest_with_features.json")
    args = parser.parse_args()

    import librosa
    import scipy.stats

    print("Loading manifest...")
    with open(args.manifest) as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    print(f"Processing {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)}...")

        # Find audio file
        audio_file = sample.get("audio_file", sample.get("labels", {}).get("file_name", ""))
        sample_id = sample.get("id", sample.get("labels", {}).get("id", ""))

        audio_path = None
        if audio_file:
            possible_paths = [
                Path(args.audio_dir) / Path(audio_file).name,
                Path(audio_file),
            ]
            for p in possible_paths:
                if p.exists():
                    audio_path = p
                    break

        if not audio_path and sample_id:
            for ext in [".wav", ".flac", ".mp3"]:
                test_path = Path(args.audio_dir) / f"sample_{int(sample_id):06d}{ext}"
                if test_path.exists():
                    audio_path = test_path
                    break

        if audio_path:
            try:
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
                features = extract_112d_features(audio, sr)
                sample["features"] = features.tolist()
            except Exception as e:
                print(f"  Error on sample {i}: {e}", file=sys.stderr)
                sample["features"] = [0.0] * 112
        else:
            sample["features"] = [0.0] * 112

    print(f"Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(manifest, f)

    # Verify
    with_features = sum(
        1
        for s in manifest["samples"]
        if s.get("features") and any(v != 0 for v in s.get("features", []))
    )
    print(f"Done! {with_features}/{len(samples)} samples have non-zero features")


if __name__ == "__main__":
    main()
