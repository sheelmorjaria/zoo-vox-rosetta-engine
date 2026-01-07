#!/usr/bin/env python3
"""
t-SNE Congruence Validation for Dynamic Microharmonic Synthesis
================================================================

This script validates that the Dynamic Microharmonic synthesizer produces
vocalizations that are statistically congruent with natural recordings.

It compares three groups:
1. Natural: Real recordings from the database
2. Concatenative: Real recordings stitched together (ground truth for naturalness)
3. Dynamic Microharmonic: Synthesized from scratch using micro-dynamics

The t-SNE plot visualizes "Acoustic Distance" - if Dynamic Micro intermingles
with Natural and Concatenative, the synthesis is statistically successful.

Usage:
    python tsne_congruence_validation.py

Output:
    - t-SNE visualization (PNG)
    - Statistical analysis
    - Congruence score
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
MARMOSET_AUDIO_INDEX = "/home/sheel/birdsong_analysis/src/audio_library/audio_index.json"
BAT_AUDIO_INDEX = "/home/sheel/birdsong_analysis/src/audio_library/bat_audio_index.json"
MICRO_DYNAMICS_DB = "/home/sheel/birdsong_analysis/src/micro_dynamics_database.json"
OUTPUT_DIR = "/home/sheel/birdsong_analysis/src/validation_results"
NUM_SAMPLES_PER_GROUP = 100  # Number of samples per group for t-SNE


def extract_micro_dynamics_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract micro-dynamics features from audio for t-SNE comparison.

    Features extracted:
    1. Attack time (ms)
    2. Decay time (ms)
    3. Vibrato rate (Hz)
    4. Vibrato depth (cents)
    5. Jitter amount
    6. Spectral flatness
    7. HNR (dB)
    8. Spectral centroid (Hz)
    9. RMS amplitude
    10. Zero-crossing rate
    """
    features = []

    # Time-domain features
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio))))

    # Spectral features
    from scipy.signal import spectrogram

    freqs, times, Sxx = spectrogram(audio, sample_rate)

    # Spectral centroid
    centroid = np.sum(freqs[:, None] * Sxx, axis=0) / np.sum(Sxx, axis=0)
    centroid_mean = np.mean(centroid)

    # Spectral flatness
    geometric_mean = np.exp(np.mean(np.log(Sxx + 1e-10), axis=0))
    arithmetic_mean = np.mean(Sxx, axis=0)
    flatness = np.mean(geometric_mean / (arithmetic_mean + 1e-10))

    # Attack time (time to reach 90% of max amplitude)
    envelope = np.abs(audio)
    max_env = np.max(envelope)
    threshold = 0.9 * max_env
    above_threshold = envelope > threshold
    if np.any(above_threshold):
        attack_sample = np.argmax(above_threshold)
        attack_ms = attack_sample / sample_rate * 1000
    else:
        attack_ms = 0

    # Decay time (time to fall to 10% of max amplitude)
    below_threshold = envelope < (0.1 * max_env)
    peak_sample = np.argmax(envelope)
    decay_samples = np.where(below_threshold[peak_sample:])[0]
    if len(decay_samples) > 0:
        decay_ms = decay_samples[0] / sample_rate * 1000
    else:
        decay_ms = len(audio) / sample_rate * 1000

    # Estimate vibrato (simplified)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(envelope, distance=int(sample_rate * 0.05))
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / sample_rate
        vibrato_rate = 1.0 / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0
    else:
        vibrato_rate = 0

    # Placeholder for vibrato depth and jitter (would need pitch tracking)
    vibrato_depth = 0
    jitter = 0

    # HNR (simplified)
    signal_energy = np.sum(audio**2)
    noise_estimate = np.diff(audio)
    noise_energy = np.sum(noise_estimate**2)
    hnr = 10 * np.log10((signal_energy + 1e-10) / (noise_energy + 1e-10))

    features = [
        attack_ms,
        decay_ms,
        vibrato_rate,
        vibrato_depth,
        jitter,
        flatness,
        hnr,
        centroid_mean,
        rms,
        zcr,
    ]

    return np.array(features)


def load_natural_samples(audio_index_path: str, num_samples: int) -> Tuple[np.ndarray, List[str]]:
    """Load natural audio samples from the audio library."""
    print(f"Loading natural samples from {audio_index_path}...")

    with open(audio_index_path, "r") as f:
        audio_index = json.load(f)

    features_list = []
    file_paths = []

    # Sample from phrase types
    phrase_keys = list(audio_index["phrases"].keys())
    max(1, num_samples // len(phrase_keys))

    for phrase_key in phrase_keys[:num_samples]:
        phrase_data = audio_index["phrases"][phrase_key]
        segments = phrase_data["segments"]

        if not segments:
            continue

        # Load a random segment
        import random

        segment = random.choice(segments)

        # Construct file path
        relative_path = segment["relative_path"]
        file_path = Path(audio_index_path).parent / relative_path

        if file_path.exists():
            try:
                audio, sr = sf.read(str(file_path))

                # Extract features
                features = extract_micro_dynamics_features(audio, sr)
                features_list.append(features)
                file_paths.append(str(file_path))
            except Exception:
                continue

        if len(features_list) >= num_samples:
            break

    print(f"  Loaded {len(features_list)} samples")

    return np.array(features_list), file_paths


def generate_concatenative_samples(
    audio_index_path: str, num_samples: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate concatenative samples by loading 2-3 phrase segments and
    concatenating them (simulating sentence-level synthesis).
    """
    print("Generating concatenative samples...")

    with open(audio_index_path, "r") as f:
        audio_index = json.load(f)

    features_list = []
    descriptions = []

    phrase_keys = list(audio_index["phrases"].keys())
    import random

    for _ in range(num_samples):
        # Select 2-3 random phrases
        num_phrases = random.randint(2, 3)
        selected_phrases = random.sample(phrase_keys, num_phrases)

        concatenated_audio = []
        total_duration = 0

        for phrase_key in selected_phrases:
            segments = audio_index["phrases"][phrase_key]["segments"]
            if segments:
                segment = random.choice(segments)
                relative_path = segment["relative_path"]
                file_path = Path(audio_index_path).parent / relative_path

                if file_path.exists():
                    try:
                        audio, sr = sf.read(str(file_path))
                        concatenated_audio.append(audio)
                        total_duration += len(audio)
                    except:
                        continue

        if concatenated_audio:
            # Concatenate
            combined = np.concatenate(concatenated_audio)

            # Extract features from concatenated sample
            features = extract_micro_dynamics_features(combined, 22050)
            features_list.append(features)
            descriptions.append(f"Concatenated_{num_phrases}_phrases")

    print(f"  Generated {len(features_list)} samples")

    return np.array(features_list), descriptions


def generate_dynamic_microharmonic_samples(
    num_samples: int, micro_dynamics_db: Dict, use_rust: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate Dynamic Microharmonic samples using the Rust synthesizer
    with real micro-dynamics parameters from the database.
    """
    print("Generating Dynamic Microharmonic samples (Rust synthesizer with real parameters)...")

    # Import Rust synthesizer
    if use_rust:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "technical_architecture",
                "/mnt/c/Users/sheel/Desktop/src/technical_architecture/target/release/libtechnical_architecture.so",
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["technical_architecture"] = module
            spec.loader.exec_module(module)
            DynamicMicroharmonicSynthesizer = module.DynamicMicroharmonicSynthesizer
            synthesizer = DynamicMicroharmonicSynthesizer(sample_rate=22050)
            print("  ✅ Using Rust DynamicMicroharmonicSynthesizer")
        except Exception as e:
            print(f"  ⚠️  Could not import Rust synthesizer: {e}")
            print("  Falling back to Python synthesis")
            use_rust = False

    features_list = []
    descriptions = []

    # Get phrase keys from micro-dynamics database
    phrase_keys = list(micro_dynamics_db.keys())

    import random

    for i in range(num_samples):
        # Select a random phrase from the micro-dynamics database
        phrase_key = random.choice(phrase_keys)
        params = micro_dynamics_db[phrase_key]

        if use_rust:
            # Use Rust synthesizer with real parameters
            try:
                # Convert spectral_flatness to spectral_tilt
                # Spectral flatness (0-1): lower = more tonal, higher = more noisy
                # Spectral tilt (dB/octave): negative values = high-freq rolloff
                # Map: flatness 0.0 → tilt -3 dB/octave, flatness 0.5 → tilt -12 dB/octave
                spectral_flatness = params.get("spectral_flatness", 0.2)
                spectral_tilt = -3.0 - (spectral_flatness * 18.0)  # Range: -3 to -12 dB/octave

                # Use real HNR value from extraction (0-10 dB typical for noisy recordings)
                hnr_db = params.get("hnr_db", 5.0)

                # Estimate shimmer from jitter (typically shimmer ≈ 0.5 * jitter)
                shimmer_amount = params["jitter"] * 0.5

                audio_list = synthesizer.synthesize_phrase(
                    f0_base=params["f0_mean"],
                    duration_ms=params["duration_ms"],
                    attack_ms=params["attack_ms"],
                    decay_ms=params["decay_ms"],
                    sustain_level=params.get("sustain_level", 0.7),
                    vibrato_rate_hz=params["vibrato_rate_hz"],
                    vibrato_depth_cents=params["vibrato_depth_cents"],
                    jitter_amount=params["jitter"],
                    shimmer_amount=shimmer_amount,
                    spectral_tilt=spectral_tilt,
                    hnr_db=hnr_db,
                )
                audio = np.array(audio_list, dtype=np.float32)
            except Exception as e:
                print(f"    ⚠️  Error synthesizing phrase {phrase_key}: {e}")
                continue
        else:
            # Fallback to Python synthesis
            sample_rate = 22050
            duration_sec = params["duration_ms"] / 1000.0
            num_samples_audio = int(duration_sec * sample_rate)

            t = np.linspace(0, duration_sec, num_samples_audio)

            # Generate with vibrato
            vibrato_osc = np.sin(2 * np.pi * params["vibrato_rate_hz"] * t)
            vibrato_cents = vibrato_osc * params["vibrato_depth_cents"]
            vibrato_ratio = 2.0 ** (vibrato_cents / 1200.0)

            # Apply jitter
            jitter = np.random.normal(0, params["jitter"], num_samples_audio)

            # Generate audio
            phase = 2 * np.pi * params["f0_mean"] * vibrato_ratio * t + jitter
            audio = np.sin(phase)

            # Apply ADSR envelope
            envelope = np.zeros_like(t)
            attack_samples = int(params["attack_ms"] / 1000 * sample_rate)
            decay_samples = int(params["decay_ms"] / 1000 * sample_rate)

            # Attack
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 3

            # Sustain
            sustain_end = len(envelope) - decay_samples
            envelope[attack_samples:sustain_end] = params.get("sustain_level", 0.7)

            # Decay
            if decay_samples > 0:
                envelope[sustain_end:] = (
                    np.linspace(params.get("sustain_level", 0.7), 0, decay_samples) ** 0.5
                )

            audio = audio.astype(np.float32) * envelope

        # Extract features from synthesized audio
        features = extract_micro_dynamics_features(audio, 22050)
        features_list.append(features)
        descriptions.append(f"Dynamic_{phrase_key}_F0_{params['f0_mean']:.0f}")

        if (i + 1) % 20 == 0:
            print(f"    Synthesized {i + 1}/{num_samples} samples...")

    print(f"  ✅ Generated {len(features_list)} samples")

    return np.array(features_list), descriptions


def run_tsne_validation(features_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run t-SNE dimensionality reduction on features.

    Returns:
        - projections: (N, 2) array of t-SNE coordinates
        - labels: (N,) array of group labels
    """
    print("\nRunning t-SNE dimensionality reduction...")

    # Combine all features
    all_features = []
    labels = []

    for group_name, features in features_dict.items():
        all_features.append(features)
        labels.extend([group_name] * len(features))

    all_features = np.vstack(all_features)
    labels = np.array(labels)

    print(f"  Total samples: {len(all_features)}")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    # Run t-SNE
    print("  Computing t-SNE projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
    projections = tsne.fit_transform(features_scaled)

    return projections, labels


def plot_tsne_results(projections: np.ndarray, labels: np.ndarray, output_path: str):
    """Plot t-SNE results and save to file."""
    print("\nPlotting t-SNE results...")

    # Color mapping
    color_map = {"Natural": "green", "Concatenative": "blue", "Dynamic_Microharmonic": "red"}

    marker_map = {"Natural": "o", "Concatenative": "o", "Dynamic_Microharmonic": "x"}

    plt.figure(figsize=(12, 9))

    for group in np.unique(labels):
        mask = labels == group
        plt.scatter(
            projections[mask, 0],
            projections[mask, 1],
            c=color_map.get(group, "gray"),
            marker=marker_map.get(group, "o"),
            label=group,
            alpha=0.6 if group != "Dynamic_Microharmonic" else 0.8,
            s=100 if group == "Dynamic_Microharmonic" else 60,
        )

    plt.legend(fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title("t-SNE Validation: Dynamic Microharmonic vs Natural vs Concatenative", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"  Saved plot to {output_path}")


def calculate_congruence_metrics(projections: np.ndarray, labels: np.ndarray) -> Dict:
    """Calculate statistical metrics of congruence."""
    print("\nCalculating congruence metrics...")

    metrics = {}

    # Get projections for each group
    natural_idx = labels == "Natural"
    concat_idx = labels == "Concatenative"
    dynamic_idx = labels == "Dynamic_Microharmonic"

    natural_proj = projections[natural_idx]
    concat_proj = projections[concat_idx]
    dynamic_proj = projections[dynamic_idx]

    # Calculate centroid distances
    natural_centroid = np.mean(natural_proj, axis=0)
    concat_centroid = np.mean(concat_proj, axis=0)
    dynamic_centroid = np.mean(dynamic_proj, axis=0)

    # Distances between centroids
    natural_concat_dist = np.linalg.norm(natural_centroid - concat_centroid)
    natural_dynamic_dist = np.linalg.norm(natural_centroid - dynamic_centroid)
    concat_dynamic_dist = np.linalg.norm(concat_centroid - dynamic_centroid)

    metrics["centroid_distances"] = {
        "natural_concatenative": float(natural_concat_dist),
        "natural_dynamic": float(natural_dynamic_dist),
        "concatenative_dynamic": float(concat_dynamic_dist),
    }

    # Calculate spread (standard deviation from centroid)
    natural_spread = np.mean(np.linalg.norm(natural_proj - natural_centroid, axis=1))
    concat_spread = np.mean(np.linalg.norm(concat_proj - concat_centroid, axis=1))
    dynamic_spread = np.mean(np.linalg.norm(dynamic_proj - dynamic_centroid, axis=1))

    metrics["spread"] = {
        "natural": float(natural_spread),
        "concatenative": float(concat_spread),
        "dynamic": float(dynamic_spread),
    }

    # Congruence score: lower distance from natural = higher congruence
    # Normalize by natural spread
    congruence_score = 1.0 / (1.0 + natural_dynamic_dist / natural_spread)
    metrics["congruence_score"] = float(congruence_score)

    print("\n  Centroid Distances:")
    print(f"    Natural ↔ Concatenative: {natural_concat_dist:.3f}")
    print(f"    Natural ↔ Dynamic: {natural_dynamic_dist:.3f}")
    print(f"    Concatenative ↔ Dynamic: {concat_dynamic_dist:.3f}")

    print("\n  Spread (within-group variance):")
    print(f"    Natural: {natural_spread:.3f}")
    print(f"    Concatenative: {concat_spread:.3f}")
    print(f"    Dynamic: {dynamic_spread:.3f}")

    print(f"\n  Congruence Score: {congruence_score:.3f}")
    print("    (1.0 = perfect congruence with natural, 0.0 = no congruence)")

    # Interpretation
    if congruence_score > 0.7:
        interpretation = "EXCELLENT: Dynamic Micro is statistically congruent with Natural"
    elif congruence_score > 0.5:
        interpretation = "GOOD: Dynamic Micro shows reasonable congruence with Natural"
    elif congruence_score > 0.3:
        interpretation = "MODERATE: Dynamic Micro shows partial congruence, needs tuning"
    else:
        interpretation = "POOR: Dynamic Micro is not congruent with Natural recordings"

    metrics["interpretation"] = interpretation
    print(f"\n  Interpretation: {interpretation}")

    return metrics


def main():
    """Main validation function."""
    print("=" * 80)
    print("t-SNE CONGRUENCE VALIDATION FOR DYNAMIC MICROHARMONIC SYNTHESIS")
    print("=" * 80)

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load micro-dynamics database
    print(f"\n📊 Loading micro-dynamics database from {MICRO_DYNAMICS_DB}...")
    with open(MICRO_DYNAMICS_DB, "r") as f:
        micro_dynamics_data = json.load(f)

    # Combine marmoset and bat micro-dynamics
    all_micro_dynamics = {
        **micro_dynamics_data["species_data"]["marmoset"],
        **micro_dynamics_data["species_data"]["egyptian_bat"],
    }

    # Filter out phrases with invalid F0 (f0_mean must be > 1000 Hz)
    micro_dynamics_db = {
        k: v
        for k, v in all_micro_dynamics.items()
        if v["f0_mean"] > 1000  # Minimum valid F0 threshold
    }

    print(
        f"  ✅ Loaded {len(micro_dynamics_db)} phrase types with valid F0 "
        f"(filtered from {len(all_micro_dynamics)} total)"
    )

    # Load samples
    features_dict = {}

    # Natural samples
    print("\n" + "=" * 80)
    natural_features, _ = load_natural_samples(MARMOSET_AUDIO_INDEX, NUM_SAMPLES_PER_GROUP)
    features_dict["Natural"] = natural_features

    # Concatenative samples
    print("\n" + "=" * 80)
    concat_features, _ = generate_concatenative_samples(MARMOSET_AUDIO_INDEX, NUM_SAMPLES_PER_GROUP)
    features_dict["Concatenative"] = concat_features

    # Dynamic Microharmonic samples (with Rust synthesizer + real parameters)
    print("\n" + "=" * 80)
    dynamic_features, _ = generate_dynamic_microharmonic_samples(
        NUM_SAMPLES_PER_GROUP, micro_dynamics_db, use_rust=True
    )
    features_dict["Dynamic_Microharmonic"] = dynamic_features

    # Run t-SNE
    print("\n" + "=" * 80)
    projections, labels = run_tsne_validation(features_dict)

    # Plot results
    plot_path = Path(OUTPUT_DIR) / "tsne_congruence_validation.png"
    plot_tsne_results(projections, labels, str(plot_path))

    # Calculate metrics
    metrics = calculate_congruence_metrics(projections, labels)

    # Save results
    results_path = Path(OUTPUT_DIR) / "tsne_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Results saved to {results_path}")

    print("\n" + "=" * 80)
    print("✅ t-SNE VALIDATION COMPLETE!")
    print("=" * 80)

    print("\n🎯 NEXT STEPS:")
    if metrics["congruence_score"] < 0.5:
        print(f"   1. Congruence score is {metrics['congruence_score']:.3f} - needs improvement")
        print("   2. Tune micro-dynamics parameters:")
        print("      - Adjust attack/decay times")
        print("      - Modify vibrato rate/depth")
        print("      - Refine jitter amount")
        print("   3. Re-run validation")
    else:
        print(f"   1. Congruence score is {metrics['congruence_score']:.3f} - good!")
        print("   2. Proceed to Bio-Acoustic Turing Test")
        print("   3. Test with live animals")

    print("=" * 80)


if __name__ == "__main__":
    main()
