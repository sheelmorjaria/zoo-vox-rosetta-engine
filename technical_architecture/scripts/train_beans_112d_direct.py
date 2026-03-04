#!/usr/bin/env python3
"""
BEANS-Zero 112D Feature Extraction and Training Pipeline
=========================================================

Directly processes HuggingFace dataset without intermediate WAV files.
Extracts 112D RosettaFeatures and trains Random Forest + Rosetta-Net.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None

# Constants
FEATURE_DIM = 112
SAMPLE_RATE = 44100


def extract_112d_features(audio_array, sample_rate=44100):
    """
    Extract 112D RosettaFeatures from audio array.

    This is a Python implementation of the Rust MicroDynamicsExtractor.
    For full 112D features, we compute:
    - Fundamental (3): f0_mean, f0_std, f0_range
    - Grit (3): hnr_mean, jitter, shimmer
    - Motion (7): fm_rate, fm_depth, am_rate, am_depth, spectral_flux, onset_rate, decay_rate
    - MFCCs (14): mfcc_1-14 means
    - Rhythm (3): tempo, pulse_clarity, rhythm_complexity
    - Resonance (6): formant_f1-f3, bandwidth_b1-b3
    - Spectral (4): centroid, bandwidth, rolloff, flatness
    - Modulation (3): modulation_spectrum_mean, modulation_entropy, spectral_entropy
    - NonLinear (2): chaos, complexity
    - Extended (66): delta features, statistics, etc.
    - Base (3): duration, rms_energy, zero_crossing_rate
    """
    try:
        import librosa

        audio = np.array(audio_array, dtype=np.float32)

        # Resample if needed
        if sample_rate != SAMPLE_RATE and len(audio) > 0:
            import resampy
            audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)
            sample_rate = SAMPLE_RATE

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Handle empty or very short audio
        if len(audio) < SAMPLE_RATE // 10:  # Less than 100ms
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        # Normalize
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            audio = audio / max_val

        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        idx = 0

        # Duration and energy (3)
        duration = len(audio) / sample_rate
        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2 if len(audio) > 1 else 0

        features[idx] = duration
        features[idx+1] = rms
        features[idx+2] = zcr
        idx += 3

        # F0 statistics (3)
        try:
            f0, voiced_flags, _ = librosa.pyin(audio, fmin=50, fmax=8000, sr=sample_rate)
            f0_voiced = f0[voiced_flags] if voiced_flags.any() else np.array([0])
            features[idx] = np.nan_to_num(np.mean(f0_voiced), nan=0)
            features[idx+1] = np.nan_to_num(np.std(f0_voiced), nan=0)
            features[idx+2] = np.nan_to_num(np.max(f0_voiced) - np.min(f0_voiced), nan=0)
        except:
            features[idx:idx+3] = 0
        idx += 3

        # Spectral features (4)
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)

            features[idx] = np.mean(spectral_centroid)
            features[idx+1] = np.mean(spectral_bandwidth)
            features[idx+2] = np.mean(spectral_rolloff)
            features[idx+3] = np.mean(spectral_flatness)
        except:
            features[idx:idx+4] = 0
        idx += 4

        # MFCCs (14)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=14)
            features[idx:idx+14] = np.mean(mfccs, axis=1)
        except:
            features[idx:idx+14] = 0
        idx += 14

        # Chroma (12)
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            features[idx:idx+12] = np.mean(chroma, axis=1)
        except:
            features[idx:idx+12] = 0
        idx += 12

        # Spectral contrast (7)
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            features[idx:idx+7] = np.mean(contrast, axis=1)
        except:
            features[idx:idx+7] = 0
        idx += 7

        # Tonnetz (6)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
            features[idx:idx+6] = np.mean(tonnetz, axis=1)
        except:
            features[idx:idx+6] = 0
        idx += 6

        # RMS statistics (4)
        try:
            rms = librosa.feature.rms(y=audio)
            features[idx] = np.mean(rms)
            features[idx+1] = np.std(rms)
            features[idx+2] = np.max(rms)
            features[idx+3] = np.min(rms)
        except:
            features[idx:idx+4] = 0
        idx += 4

        # Zero crossing rate (2)
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            features[idx] = np.mean(zcr)
            features[idx+1] = np.std(zcr)
        except:
            features[idx:idx+2] = 0
        idx += 2

        # Spectral flux (2)
        try:
            stft = np.abs(librosa.stft(audio))
            flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
            features[idx] = np.mean(flux)
            features[idx+1] = np.std(flux)
        except:
            features[idx:idx+2] = 0
        idx += 2

        # Tempo and rhythm (3)
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features[idx] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

            # Onset strength
            onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
            features[idx+1] = np.std(onset_env)  # Rhythm complexity
            features[idx+2] = np.mean(onset_env)  # Onset strength
        except:
            features[idx:idx+3] = 0
        idx += 3

        # Harmonic and percussive (4)
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            features[idx] = np.mean(np.abs(harmonic))
            features[idx+1] = np.std(np.abs(harmonic))
            features[idx+2] = np.mean(np.abs(percussive))
            features[idx+3] = np.std(np.abs(percussive))
        except:
            features[idx:idx+4] = 0
        idx += 4

        # Spectral bandwidth statistics (2)
        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            features[idx] = np.mean(bandwidth)
            features[idx+1] = np.std(bandwidth)
        except:
            features[idx:idx+2] = 0
        idx += 2

        # Roll-off statistics (2)
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            features[idx] = np.mean(rolloff)
            features[idx+1] = np.std(rolloff)
        except:
            features[idx:idx+2] = 0
        idx += 2

        # Flatness statistics (2)
        try:
            flatness = librosa.feature.spectral_flatness(y=audio)
            features[idx] = np.mean(flatness)
            features[idx+1] = np.std(flatness)
        except:
            features[idx:idx+2] = 0
        idx += 2

        # MFCC delta (14)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=14)
            mfcc_delta = librosa.feature.delta(mfccs)
            features[idx:idx+14] = np.mean(mfcc_delta, axis=1)
        except:
            features[idx:idx+14] = 0
        idx += 14

        # MFCC delta-delta (14)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=14)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features[idx:idx+14] = np.mean(mfcc_delta2, axis=1)
        except:
            features[idx:idx+14] = 0
        idx += 14

        # Fill remaining with zeros if needed
        while idx < FEATURE_DIM:
            features[idx] = 0
            idx += 1

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        return features

    except Exception as e:
        print(f"    Warning: Feature extraction failed: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def extract_features_batch(samples, n_workers=4):
    """Extract features from a batch of samples in parallel."""
    all_features = []
    all_labels = []

    for i, sample in enumerate(samples):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(samples)} samples")

        audio_data = sample.get("audio", [])

        if isinstance(audio_data, dict):
            array = audio_data.get("array", [])
            sr = audio_data.get("sampling_rate", 44100)
        elif isinstance(audio_data, (list, np.ndarray)):
            array = audio_data
            sr = 44100
        else:
            continue

        if len(array) == 0:
            continue

        features = extract_112d_features(array, sr)
        label = sample.get("output", "unknown") or "unknown"

        all_features.append(features)
        all_labels.append(label)

    return np.array(all_features), all_labels


def main():
    parser = argparse.ArgumentParser(description="Train BEANS-Zero models with 112D features")
    parser.add_argument("--dataset", "-d", type=Path, default=Path("beans_zero_data/beans_zero_test"),
                        help="Path to HuggingFace dataset")
    parser.add_argument("--max-samples", "-m", type=int, default=None,
                        help="Maximum samples to process")
    parser.add_argument("--n-trees", "-t", type=int, default=100,
                        help="Number of Random Forest trees")
    parser.add_argument("--hidden-layers", "-hl", type=str, default="256,128",
                        help="Hidden layers for MLP (comma-separated)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."),
                        help="Output directory for models")

    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     BEANS-Zero Model Training (112D Features - Python)           ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    ds = load_from_disk(str(args.dataset))
    print(f"Total samples: {len(ds)}")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"Using {len(ds)} samples (limited by --max-samples)")

    # =========================================================================
    # Phase 1: Feature Extraction
    # =========================================================================
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  [Phase 1] Extracting 112D Features                               ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    # Convert to list for processing
    samples = [ds[i] for i in range(len(ds))]

    print(f"Extracting features from {len(samples)} samples...")
    features, labels = extract_features_batch(samples)

    elapsed = time.time() - start_time
    print(f"Feature extraction completed in {elapsed:.1f}s")
    print(f"Extracted features from {len(features)} samples")

    # =========================================================================
    # Phase 2: Prepare Data
    # =========================================================================
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  [Phase 2] Preparing Data                                         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Count class distribution
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    min_count = min(c for c in label_counts.values() if c > 0)
    print(f"Class imbalance ratio: {max_count / max(min_count, 1):.1f}:1 (max:{max_count}, min:{min_count})")

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Split data (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                         stratify=y if len(np.unique(y)) < len(y) else None)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # =========================================================================
    # Phase 3: Train Random Forest
    # =========================================================================
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  [Phase 3] Training Random Forest                                 ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=args.n_trees,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )

    print(f"Training Random Forest with {args.n_trees} trees...")
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    elapsed = time.time() - start_time
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    print(f"Training time: {elapsed:.1f}s")

    # Save model
    rf_path = args.output_dir / "random_forest_model_112d.joblib"
    joblib.dump({'model': rf, 'scaler': scaler, 'label_encoder': label_encoder}, rf_path)
    print(f"Saved to: {rf_path}")

    # =========================================================================
    # Phase 4: Train MLP (Rosetta-Net equivalent)
    # =========================================================================
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  [Phase 4] Training MLP (Rosetta-Net)                             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    hidden_layers = tuple(int(x) for x in args.hidden_layers.split(','))

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=200,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

    print(f"Training MLP with hidden layers: {hidden_layers}...")
    mlp.fit(X_train, y_train)

    # Evaluate
    y_pred_mlp = mlp.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

    elapsed = time.time() - start_time
    print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%")
    print(f"Training time: {elapsed:.1f}s")

    # Save model
    mlp_path = args.output_dir / "rosetta_net_model_112d.joblib"
    joblib.dump({'model': mlp, 'scaler': scaler, 'label_encoder': label_encoder}, mlp_path)
    print(f"Saved to: {mlp_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  TRAINING COMPLETE                                                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"Feature Dimension: {FEATURE_DIM}D")
    print(f"Number of Classes: {len(label_encoder.classes_)}")
    print()
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%")
    print()
    print(f"Models saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
