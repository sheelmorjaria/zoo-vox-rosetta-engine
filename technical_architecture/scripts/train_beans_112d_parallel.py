#!/usr/bin/env python3
"""
BEANS-Zero 112D Feature Extraction and Training Pipeline
=========================================================

Directly processes HuggingFace dataset with PARALLEL feature extraction.
Uses multiprocessing for 8x faster feature extraction.
"""

import argparse
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from datasets import load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Constants
FEATURE_DIM = 112
SAMPLE_RATE = 44100


def extract_112d_features(audio_array, sample_rate=44100):
    """Extract 112D RosettaFeatures from audio array."""
    try:
        import librosa

        audio = np.array(audio_array, dtype=np.float32)

        if len(audio) < SAMPLE_RATE // 10:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        # Normalize
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            audio = audio / max_val

        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        idx = 0

        # Duration and energy (3)
        duration = len(audio) / sample_rate
        rms = np.sqrt(np.mean(audio**2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2 if len(audio) > 1 else 0
        features[idx : idx + 3] = [duration, rms, zcr]
        idx += 3

        # F0 statistics (3)
        try:
            f0, voiced_flags, _ = librosa.pyin(audio, fmin=50, fmax=8000, sr=sample_rate)
            f0_voiced = f0[voiced_flags] if voiced_flags.any() else np.array([0])
            features[idx : idx + 3] = [
                np.nan_to_num(np.mean(f0_voiced), nan=0),
                np.nan_to_num(np.std(f0_voiced), nan=0),
                np.nan_to_num(np.max(f0_voiced) - np.min(f0_voiced), nan=0),
            ]
        except:
            pass
        idx += 3

        # Spectral features (4)
        try:
            features[idx] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            features[idx + 1] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
            features[idx + 2] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
            features[idx + 3] = np.mean(librosa.feature.spectral_flatness(y=audio))
        except:
            pass
        idx += 4

        # MFCCs (14)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=14)
            features[idx : idx + 14] = np.mean(mfccs, axis=1)
        except:
            pass
        idx += 14

        # Chroma (12)
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            features[idx : idx + 12] = np.mean(chroma, axis=1)
        except:
            pass
        idx += 12

        # Spectral contrast (7)
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            features[idx : idx + 7] = np.mean(contrast, axis=1)
        except:
            pass
        idx += 7

        # Tonnetz (6)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
            features[idx : idx + 6] = np.mean(tonnetz, axis=1)
        except:
            pass
        idx += 6

        # RMS statistics (4)
        try:
            rms = librosa.feature.rms(y=audio)
            features[idx : idx + 4] = [np.mean(rms), np.std(rms), np.max(rms), np.min(rms)]
        except:
            pass
        idx += 4

        # Zero crossing rate (2)
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            features[idx : idx + 2] = [np.mean(zcr), np.std(zcr)]
        except:
            pass
        idx += 2

        # Spectral flux (2)
        try:
            stft = np.abs(librosa.stft(audio))
            flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
            features[idx : idx + 2] = [np.mean(flux), np.std(flux)]
        except:
            pass
        idx += 2

        # Tempo and rhythm (3)
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features[idx] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
            features[idx + 1] = np.std(onset_env)
            features[idx + 2] = np.mean(onset_env)
        except:
            pass
        idx += 3

        # Harmonic and percussive (4)
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            features[idx : idx + 4] = [
                np.mean(np.abs(harmonic)),
                np.std(np.abs(harmonic)),
                np.mean(np.abs(percussive)),
                np.std(np.abs(percussive)),
            ]
        except:
            pass
        idx += 4

        # More spectral stats (6)
        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            features[idx] = np.mean(bandwidth)
            features[idx + 1] = np.std(bandwidth)
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            features[idx + 2] = np.mean(rolloff)
            features[idx + 3] = np.std(rolloff)
            flatness = librosa.feature.spectral_flatness(y=audio)
            features[idx + 4] = np.mean(flatness)
            features[idx + 5] = np.std(flatness)
        except:
            pass
        idx += 6

        # MFCC delta (14)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=14)
            mfcc_delta = librosa.feature.delta(mfccs)
            features[idx : idx + 14] = np.mean(mfcc_delta, axis=1)
        except:
            pass
        idx += 14

        # MFCC delta-delta (14)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=14)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features[idx : idx + 14] = np.mean(mfcc_delta2, axis=1)
        except:
            pass
        idx += 14

        return np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    except Exception:
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def process_sample(sample):
    """Process a single sample - for parallel execution."""
    audio_data = sample.get("audio", [])

    if isinstance(audio_data, dict):
        array = audio_data.get("array", [])
        sr = audio_data.get("sampling_rate", 44100)
    elif isinstance(audio_data, (list, np.ndarray)):
        array = audio_data
        sr = 44100
    else:
        return None, None

    if len(array) == 0:
        return None, None

    features = extract_112d_features(array, sr)
    label = sample.get("output", "unknown") or "unknown"

    return features, label


def main():
    parser = argparse.ArgumentParser(description="Train BEANS-Zero models with 112D features")
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path("beans_zero_data/beans_zero_test"),
        help="Path to HuggingFace dataset",
    )
    parser.add_argument(
        "--max-samples", "-m", type=int, default=None, help="Maximum samples to process"
    )
    parser.add_argument(
        "--n-trees", "-t", type=int, default=100, help="Number of Random Forest trees"
    )
    parser.add_argument(
        "--n-workers", "-w", type=int, default=mp.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("."), help="Output directory for models"
    )

    args = parser.parse_args()

    print(flush=True)
    print("╔═══════════════════════════════════════════════════════════════════╗", flush=True)
    print("║     BEANS-Zero 112D Training (Parallel Extraction)               ║", flush=True)
    print("╚═══════════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)

    # Load dataset
    print(f"Loading dataset from: {args.dataset}", flush=True)
    ds = load_from_disk(str(args.dataset))
    print(f"Total samples: {len(ds)}", flush=True)

    n_samples = len(ds)
    if args.max_samples:
        n_samples = min(args.max_samples, len(ds))
        ds = ds.select(range(n_samples))
        print(f"Using {n_samples} samples (limited by --max-samples)", flush=True)

    # =========================================================================
    # Phase 1: Parallel Feature Extraction
    # =========================================================================
    print(flush=True)
    print("╔═══════════════════════════════════════════════════════════════════╗", flush=True)
    print(
        f"║  [Phase 1] Extracting 112D Features ({args.n_workers} workers)          ║", flush=True
    )
    print("╚═══════════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)

    start_time = time.time()

    all_features = []
    all_labels = []

    # Convert dataset to list of samples
    print(
        f"Extracting features from {n_samples} samples with {args.n_workers} workers...", flush=True
    )

    # Process in parallel using multiprocessing
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sample, ds[i]): i for i in range(n_samples)}

        completed = 0
        for future in as_completed(futures):
            features, label = future.result()
            if features is not None:
                all_features.append(features)
                all_labels.append(label)

            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (n_samples - completed) / rate
                print(
                    f"  Processed {completed}/{n_samples} ({completed / n_samples * 100:.1f}%) - "
                    f"{rate:.1f} samples/s - ETA: {eta / 60:.1f}min",
                    flush=True,
                )

    elapsed = time.time() - start_time
    print(
        f"Feature extraction completed in {elapsed:.1f}s ({n_samples / elapsed:.1f} samples/s)",
        flush=True,
    )
    print(f"Extracted features from {len(all_features)} samples", flush=True)

    if len(all_features) == 0:
        print("Error: No features extracted!", flush=True)
        return

    features = np.array(all_features)
    labels = all_labels

    # =========================================================================
    # Phase 2: Prepare Data
    # =========================================================================
    print(flush=True)
    print("╔═══════════════════════════════════════════════════════════════════╗", flush=True)
    print("║  [Phase 2] Preparing Data                                         ║", flush=True)
    print("╚═══════════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"Number of classes: {len(label_encoder.classes_)}", flush=True)

    # Count class distribution
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    min_count = min(c for c in label_counts.values() if c > 0)
    print(
        f"Class imbalance ratio: {max_count / max(min_count, 1):.1f}:1 (max:{max_count}, min:{min_count})",
        flush=True,
    )

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Split data (80/20)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) < len(y) else None
    )

    print(f"Training samples: {len(X_train)}", flush=True)
    print(f"Test samples: {len(X_test)}", flush=True)

    # =========================================================================
    # Phase 3: Train Random Forest
    # =========================================================================
    print(flush=True)
    print("╔═══════════════════════════════════════════════════════════════════╗", flush=True)
    print(f"║  [Phase 3] Training Random Forest ({args.n_trees} trees)             ║", flush=True)
    print("╚═══════════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)

    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=args.n_trees, max_depth=20, n_jobs=-1, random_state=42, class_weight="balanced"
    )

    print(f"Training Random Forest with {args.n_trees} trees...", flush=True)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    elapsed = time.time() - start_time
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%", flush=True)
    print(f"Training time: {elapsed:.1f}s", flush=True)

    # Save model
    rf_path = args.output_dir / "random_forest_model_112d.joblib"
    joblib.dump({"model": rf, "scaler": scaler, "label_encoder": label_encoder}, rf_path)
    print(f"Saved to: {rf_path}", flush=True)

    # =========================================================================
    # Phase 4: Train MLP
    # =========================================================================
    print(flush=True)
    print("╔═══════════════════════════════════════════════════════════════════╗", flush=True)
    print("║  [Phase 4] Training MLP (Rosetta-Net)                             ║", flush=True)
    print("╚═══════════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)

    start_time = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=100,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True,
    )

    print("Training MLP with hidden layers: (512, 256, 128)...", flush=True)
    mlp.fit(X_train, y_train)

    # Evaluate
    y_pred_mlp = mlp.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

    elapsed = time.time() - start_time
    print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%", flush=True)
    print(f"Training time: {elapsed:.1f}s", flush=True)

    # Save model
    mlp_path = args.output_dir / "rosetta_net_model_112d.joblib"
    joblib.dump({"model": mlp, "scaler": scaler, "label_encoder": label_encoder}, mlp_path)
    print(f"Saved to: {mlp_path}", flush=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print(flush=True)
    print("╔═══════════════════════════════════════════════════════════════════╗", flush=True)
    print("║  TRAINING COMPLETE                                                ║", flush=True)
    print("╚═══════════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)
    print(f"Feature Dimension: {FEATURE_DIM}D", flush=True)
    print(f"Number of Classes: {len(label_encoder.classes_)}", flush=True)
    print(f"Total Samples: {len(features)}", flush=True)
    print(flush=True)
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%", flush=True)
    print(f"MLP Accuracy: {mlp_accuracy * 100:.2f}%", flush=True)
    print(flush=True)
    print(f"Models saved to: {args.output_dir}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
