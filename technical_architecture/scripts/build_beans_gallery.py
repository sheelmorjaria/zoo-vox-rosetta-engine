#!/usr/bin/env python3
"""Build reference gallery from BEANS-Zero manifest and audio files.

This creates a gallery with one representative sample per unique species,
with 112D features and 64D Siamese embeddings.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def extract_112d_features(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Extract 112D features matching the Rust implementation.

    Feature stack architecture:
    - Layer 1 (0-45): Base Physics - Duration, F0, Resonance, Spectral
    - Layer 2 (46-75): Macro Texture - Harmonic Density, GLCM Roughness
    - Layer 3 (76-111): Micro Texture - FM Bins, ICI Bins, Dynamics, Rhythm
    """
    import librosa
    from scipy.stats import kurtosis, skew

    features = np.zeros(112, dtype=np.float32)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 1e-10:
        audio = audio / max_val

    # Basic temporal features
    duration_ms = len(audio) / sr * 1000.0
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0 if len(audio) > 1 else 0.0

    try:
        n_fft = min(2048, len(audio) // 2)
        hop_length = n_fft // 4

        if n_fft < 64:
            # Audio too short
            features[0] = duration_ms
            features[2] = rms
            return features

        # Compute spectrogram
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_mean = S.mean(axis=1)

        # =====================================================================
        # Layer 1: Base Physics (0-45)
        # =====================================================================

        # Duration and basic (0-5)
        features[0] = duration_ms
        features[1] = duration_ms / 1000.0  # duration_seconds
        features[2] = rms
        features[3] = zcr

        # F0 estimation (4-7)
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

        # Spectral centroid (8)
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features[8] = cent.mean()

        # Spectral bandwidth (9)
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features[9] = bw.mean()

        # Spectral rolloff (10-12)
        rolloff85 = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        )
        rolloff95 = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.95
        )
        features[10] = rolloff85.mean()
        features[11] = rolloff95.mean()
        features[12] = rolloff95.mean() - rolloff85.mean()  # rolloff_range

        # Spectral flatness (13)
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
        features[13] = np.clip(flatness.mean(), 0, 1)

        # Spectral contrast (14-20)
        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
            features[14] = contrast.mean()
            for i in range(min(6, contrast.shape[0])):
                features[15 + i] = contrast[i].mean()
        except Exception:
            pass

        # Harmonicity (21-23)
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            features[21] = np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio)) + 1e-10)
            features[22] = np.mean(np.abs(percussive)) / (np.mean(np.abs(audio)) + 1e-10)
            features[23] = features[21] / (features[22] + 1e-10)  # HNR proxy
        except Exception:
            pass

        # Attack/Decay time (24-27)
        envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sr))
        if len(envelope) > 0:
            peak_idx = np.argmax(envelope)
            peak_val = envelope[peak_idx]

            # Attack time (to 90% of peak)
            attack_idx = (
                np.argmax(envelope[: peak_idx + 1] >= 0.9 * peak_val) if peak_idx > 0 else 0
            )
            features[24] = (attack_idx * hop_length / sr) * 1000.0  # ms

            # Decay time (from peak to 10%)
            if peak_idx < len(envelope) - 1:
                decay_envelope = envelope[peak_idx:]
                decay_idx = np.argmax(decay_envelope < 0.1 * peak_val)
                if decay_idx > 0:
                    features[25] = (decay_idx * hop_length / sr) * 1000.0  # ms

            features[26] = peak_val
            features[27] = np.std(envelope)

        # MFCCs (28-40)
        try:
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
            )
            for i in range(min(13, mfcc.shape[0])):
                features[28 + i] = mfcc[i].mean()
        except Exception:
            pass

        # Energy stats (41-45)
        rms_feat = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
        features[41] = rms_feat.mean()
        features[42] = np.std(rms_feat) if len(rms_feat) > 1 else 0
        features[43] = skew(audio)
        features[44] = kurtosis(audio)
        features[45] = np.percentile(np.abs(audio), 95)

        # =====================================================================
        # Layer 2: Macro Texture (46-75) - Harmonic Density, GLCM
        # =====================================================================

        # Chroma (46-57)
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
            for i in range(min(12, chroma.shape[0])):
                features[46 + i] = chroma[i].mean()
        except Exception:
            pass

        # Tonnetz (58-63)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            for i in range(min(6, tonnetz.shape[0])):
                features[58 + i] = tonnetz[i].mean()
        except Exception:
            pass

        # Spectral flux (64-66)
        delta = np.diff(S_mean)
        features[64] = np.mean(np.abs(delta))
        features[65] = np.std(delta) if len(delta) > 1 else 0
        features[66] = np.max(np.abs(delta)) if len(delta) > 0 else 0

        # Band energy ratios (67-75)
        n_bands = 9
        band_size = S.shape[0] // n_bands
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            band_energy = np.mean(S_mean[start:end]) if end <= len(S_mean) else 0
            features[67 + i] = band_energy

        # =====================================================================
        # Layer 3: Micro Texture (76-111) - FM, ICI, Dynamics, Rhythm
        # =====================================================================

        # Tempo and rhythm (76-78)
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features[76] = (
                float(tempo) if np.isscalar(tempo) else (float(tempo[0]) if len(tempo) > 0 else 0.0)
            )

            # Onset rate
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            features[77] = len(onsets) / (duration_ms / 1000.0) if duration_ms > 0 else 0

            # ICI (Inter-onset interval) stats
            if len(onsets) > 1:
                ici = np.diff(onsets) * hop_length / sr * 1000  # ms
                features[78] = np.mean(ici)
                features[79] = np.std(ici)
                features[80] = np.median(ici)
        except Exception:
            pass

        # Vibrato/tremolo detection (81-85)
        try:
            # Amplitude modulation
            amp_env = librosa.effects.hpss(audio)[0]  # harmonic component
            if len(amp_env) > sr // 10:  # Need at least 100ms
                # FFT of amplitude envelope
                amp_fft = np.abs(np.fft.rfft(np.abs(amp_env)))
                freqs = np.fft.rfftfreq(len(amp_env), 1 / sr)
                # Find peaks in 4-8 Hz range (vibrato)
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

        # Spectral variability (86-92)
        try:
            spec_var = np.std(S, axis=1)
            features[86] = np.mean(spec_var)
            features[87] = np.std(spec_var)
            features[88] = np.max(spec_var)

            # Spectral skewness and kurtosis per frame
            spec_skew = np.array([skew(S[:, i]) for i in range(S.shape[1])])
            spec_kurt = np.array([kurtosis(S[:, i]) for i in range(S.shape[1])])
            features[89] = np.mean(spec_skew)
            features[90] = np.mean(spec_kurt)
            features[91] = np.std(spec_skew)
            features[92] = np.std(spec_kurt)
        except Exception:
            pass

        # Dynamics (93-100)
        features[93] = np.max(audio) - np.min(audio)  # dynamic range
        features[94] = np.percentile(audio, 90) - np.percentile(audio, 10)  # inter-quartile range

        # Zero crossing variability
        try:
            zcr_frames = librosa.feature.zero_crossing_rate(
                audio, frame_length=n_fft, hop_length=hop_length
            )
            features[95] = np.std(zcr_frames)
            features[96] = np.percentile(zcr_frames, 90)
        except Exception:
            pass

        # Spectral entropy (97-99)
        try:
            S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
            entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
            features[97] = np.mean(entropy)
            features[98] = np.std(entropy)
            features[99] = np.min(entropy)
        except Exception:
            pass

        # Final features (100-111) - additional spectral stats
        try:
            # Spectral slope
            freqs_spec = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            spec_slope = np.polyfit(freqs_spec[: len(S_mean)], S_mean, 1)
            features[100] = spec_slope[0]  # slope
            features[101] = spec_slope[1]  # intercept

            # Spectral decrease
            features[102] = np.mean(np.diff(S_mean) / (freqs_spec[1 : len(S_mean) + 1] + 1e-10))

            # Spectral spread
            features[103] = np.sqrt(
                np.sum(((freqs_spec[: len(S_mean)] - features[8]) ** 2) * S_mean)
                / (S_mean.sum() + 1e-10)
            )

            # Spectral irregularity
            features[104] = np.sum(np.abs(np.diff(S_mean))) / (S_mean.sum() + 1e-10)

            # Additional MFCC deltas
            if mfcc.shape[1] > 1:
                mfcc_delta = np.diff(mfcc, axis=1)
                features[105] = np.mean(np.abs(mfcc_delta))
                features[106] = np.std(mfcc_delta)

            # Harmonic ratio
            features[107] = features[21] if features[21] > 0 else 0.1

            # Energy in frequency bands (108-111)
            low_band = S_mean[: len(S_mean) // 4]
            mid_band = S_mean[len(S_mean) // 4 : len(S_mean) // 2]
            high_band = S_mean[len(S_mean) // 2 :]
            features[108] = np.sum(low_band) / (np.sum(S_mean) + 1e-10)
            features[109] = np.sum(mid_band) / (np.sum(S_mean) + 1e-10)
            features[110] = np.sum(high_band) / (np.sum(S_mean) + 1e-10)
            features[111] = features[108] / (features[110] + 1e-10)  # low/high ratio

        except Exception:
            pass

    except Exception as e:
        print(f"Warning: Feature extraction failed: {e}", file=sys.stderr)
        features[0] = duration_ms
        features[2] = rms

    # Clean up any NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def generate_siamese_embedding(features: np.ndarray, latent_dim: int = 64) -> np.ndarray:
    """
    Generate a simple Siamese embedding using a deterministic transform.

    This is a simplified version - in production, use trained weights.
    Uses a pseudo-random projection for dimensionality reduction.
    """
    # Normalize features
    norm = np.linalg.norm(features)
    if norm > 1e-10:
        features = features / norm

    # Simple hash-based projection (deterministic)
    np.random.seed(42)  # Fixed seed for reproducibility
    projection = np.random.randn(112, latent_dim).astype(np.float32)
    projection /= np.sqrt(112)  # Scale

    embedding = np.maximum(0, np.dot(features, projection))  # ReLU activation

    # Normalize embedding
    norm = np.linalg.norm(embedding)
    if norm > 1e-10:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def map_species_to_taxon(species: str) -> str:
    """
    Map species name to taxonomic group.

    Must match Rust Taxon enum:
    Cetacean, Mysticete, Songbird, NonPasserine, Amphibian, Pinniped, Insect, Mammal, Unknown
    """
    species_lower = species.lower()

    # Non-passerine birds (parrots, owls, woodpeckers, hummingbirds, doves, etc.)
    non_passerine_keywords = [
        "parrot",
        "owl",
        "woodpecker",
        "hummingbird",
        "dove",
        "pigeon",
        "swift",
        "nightjar",
        "whip-poor-will",
        "kingfisher",
        "heron",
        "egret",
        "gull",
        "tern",
        "sandpiper",
        "plover",
        "falcon",
        "eagle",
        "hawk",
        "kookaburra",
        "psittaciformes",
        "strigiformes",
    ]

    # Songbirds (passerines)
    songbird_keywords = [
        "finch",
        "sparrow",
        "robin",
        "wren",
        "thrush",
        "warbler",
        "cardinal",
        "jay",
        "mockingbird",
        "blackbird",
        "chickadee",
        "titmouse",
        "nuthatch",
        "creeper",
        "starling",
        "swallow",
        "martin",
        "flycatcher",
        "vireo",
        "oriole",
        "tanager",
        "bunting",
        "grosbeak",
        "cowbird",
        "meadowlark",
        "lark",
        "passeriformes",
        "timaliidae",
        "muscicapidae",
        "sylviidae",
    ]

    # Toothed whales (dolphins, porpoises) - clicks and whistles
    cetacean_keywords = [
        "dolphin",
        "porpoise",
        "orca",
        "killer whale",
        "phocoenidae",
        "delphinidae",
    ]

    # Baleen whales (humpback, blue, minke) - songs and moans
    mysticete_keywords = [
        "humpback",
        "blue whale",
        "minke",
        "right whale",
        "gray whale",
        "fin whale",
        "sei whale",
        "mysticeti",
        "balaenidae",
        "rorqual",
    ]

    # Pinnipeds (seals, sea lions, walruses)
    pinniped_keywords = ["seal", "sea lion", "walrus", "pinniped", "otariidae", "phocidae"]

    # Bats and terrestrial mammals
    bat_keywords = ["bat", "echolocat"]
    mammal_keywords = [
        "gibbon",
        "monkey",
        "primate",
        "meerkat",
        "hyena",
        "squirrel",
        "deer",
        "pig",
        "boar",
        "artiodactyla",
        "primates",
    ]

    # Amphibians
    amphibian_keywords = [
        "frog",
        "toad",
        "peeper",
        "tree frog",
        "chorus frog",
        "anura",
        "amphibia",
        "alytidae",
        "dicroglossidae",
    ]

    # Insects
    insect_keywords = [
        "cricket",
        "cicada",
        "mosquito",
        "insect",
        "orthoptera",
        "katydid",
        "grasshopper",
        "beetle",
        "diptera",
        "coleoptera",
    ]

    # Check whales first (more specific)
    for kw in mysticete_keywords:
        if kw in species_lower:
            return "Mysticete"

    for kw in cetacean_keywords:
        if kw in species_lower:
            return "Cetacean"

    # Generic whale check
    if "whale" in species_lower:
        # Minke and Sperm are toothed whales
        if "minke" in species_lower or "sperm" in species_lower:
            return "Cetacean"
        # Default to Mysticete for unknown whales
        return "Mysticete"

    for kw in pinniped_keywords:
        if kw in species_lower:
            return "Pinniped"

    for kw in bat_keywords:
        if kw in species_lower:
            return "Mammal"

    # Check birds - need to distinguish songbird vs non-passerine
    for kw in songbird_keywords:
        if kw in species_lower:
            return "Songbird"

    for kw in non_passerine_keywords:
        if kw in species_lower:
            return "NonPasserine"

    for kw in amphibian_keywords:
        if kw in species_lower:
            return "Amphibian"

    for kw in insect_keywords:
        if kw in species_lower:
            return "Insect"

    for kw in mammal_keywords:
        if kw in species_lower:
            return "Mammal"

    # Check for scientific names
    if "passeriformes" in species_lower:
        return "Songbird"
    if "aves" in species_lower:
        return "Songbird"  # Default for birds
    if "cetacea" in species_lower or "phocoenidae" in species_lower:
        return "Cetacean"
    if "amphibia" in species_lower or "anura" in species_lower:
        return "Amphibian"
    if "insecta" in species_lower or "orthoptera" in species_lower:
        return "Insect"
    if "mammalia" in species_lower:
        return "Mammal"

    return "Unknown"


def extract_label(sample: dict) -> str:
    """Extract the species label from a manifest sample."""
    labels = sample.get("labels", {})

    # Try 'output' first (for captioning tasks)
    output = labels.get("output")
    if output and output != "None":
        return output

    # Try 'metadata'
    metadata = labels.get("metadata")
    if metadata and metadata != "None":
        return metadata

    # Fall back to source_dataset
    source = labels.get("source_dataset", "Unknown")
    return source


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build reference gallery from BEANS-Zero")
    parser.add_argument(
        "--manifest", default="beans_zero_manifest_compat.json", help="Path to manifest JSON"
    )
    parser.add_argument(
        "--audio-dir", default="beans_audio_full_rust", help="Directory containing audio files"
    )
    parser.add_argument(
        "--output", default="beans_reference_gallery.json", help="Output gallery JSON path"
    )
    parser.add_argument(
        "--samples-per-species",
        type=int,
        default=1,
        help="Number of samples per species to include",
    )
    parser.add_argument(
        "--min-samples", type=int, default=2, help="Minimum samples for species to be included"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BEANS-Zero Reference Gallery Builder")
    print("=" * 70)

    # Load manifest
    print(f"\n[1] Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    print(f"    Total samples in manifest: {len(samples)}")

    # Group samples by species
    species_samples = defaultdict(list)
    for sample in samples:
        label = extract_label(sample)
        species_samples[label].append(sample)

    print(f"    Unique labels: {len(species_samples)}")

    # Filter species with minimum samples
    valid_species = {
        s: samples for s, samples in species_samples.items() if len(samples) >= args.min_samples
    }
    print(f"    Species with >= {args.min_samples} samples: {len(valid_species)}")

    # Build gallery
    print(f"\n[2] Building gallery from audio files...")
    print(f"    Audio directory: {args.audio_dir}")
    print(f"    Samples per species: {args.samples_per_species}")

    gallery = {"samples": []}
    processed = 0
    skipped = 0

    import librosa

    for species, species_sample_list in sorted(valid_species.items()):
        # Take first N samples for this species
        for sample in species_sample_list[: args.samples_per_species]:
            # Find audio file
            audio_file = sample.get("audio_file", sample.get("labels", {}).get("file_name", ""))
            if not audio_file:
                skipped += 1
                continue

            # Try multiple possible paths
            possible_paths = [
                Path(args.audio_dir) / Path(audio_file).name,
                Path(audio_file),
                Path(args.audio_dir) / audio_file,
            ]

            audio_path = None
            for p in possible_paths:
                if p.exists():
                    audio_path = p
                    break

            if not audio_path:
                # Try with sample ID
                sample_id = sample.get("id", sample.get("labels", {}).get("id", ""))
                if sample_id:
                    # Try various naming conventions
                    for ext in [".wav", ".flac", ".mp3"]:
                        test_path = Path(args.audio_dir) / f"sample_{int(sample_id):06d}{ext}"
                        if test_path.exists():
                            audio_path = test_path
                            break

            if not audio_path:
                skipped += 1
                continue

            try:
                # Load audio
                audio, sr = librosa.load(audio_path, sr=None, mono=True)

                # Extract features
                features = extract_112d_features(audio, sr)

                # For zero-shot, use raw features directly as "embeddings"
                # This allows k-NN on the 112D feature space
                embedding = features.tolist()  # Use raw features as embeddings

                # Determine taxon
                taxon = map_species_to_taxon(species)

                # Add to gallery
                gallery["samples"].append(
                    {
                        "species": species,
                        "taxon": taxon,
                        "embedding": embedding.tolist(),
                        "original_features": features.tolist(),
                    }
                )

                processed += 1
                if processed % 50 == 0:
                    print(f"    Processed {processed} species...")

            except Exception as e:
                print(f"    Error processing {audio_path}: {e}", file=sys.stderr)
                skipped += 1

    print(f"\n[3] Gallery Statistics:")
    print(f"    Successfully processed: {processed}")
    print(f"    Skipped: {skipped}")
    print(f"    Gallery size: {len(gallery['samples'])}")

    # Taxon distribution
    taxon_counts = defaultdict(int)
    for s in gallery["samples"]:
        taxon_counts[s["taxon"]] += 1

    print(f"\n    Taxon distribution:")
    for taxon, count in sorted(taxon_counts.items(), key=lambda x: -x[1]):
        print(f"      {taxon}: {count}")

    # Save gallery
    print(f"\n[4] Saving gallery to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(gallery, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Gallery building complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
