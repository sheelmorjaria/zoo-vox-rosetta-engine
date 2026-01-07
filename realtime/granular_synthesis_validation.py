#!/usr/bin/env python3
"""
Granular Concatenative Synthesis Validation
============================================

This script validates that Granular Concatenative Synthesis achieves
high-fidelity results (t-SNE distance < 7.0) by preserving formant structure.

Hypothesis:
- Concatenative synthesis: distance ≈ 4.2 (baseline, real audio)
- Additive synthesis: distance ≈ 27.0 (poor, no formant structure)
- Granular synthesis: distance < 7.0 (good, preserves formants)

Usage:
    python granular_synthesis_validation.py

Output:
    - t-SNE visualization comparing Natural, Concatenative, and Granular
    - Statistical analysis
    - Congruence scores
"""

import json
import random
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
MARMOSET_AUDIO_INDEX = '/home/sheel/birdsong_analysis/src/audio_library/audio_index.json'
OUTPUT_DIR = '/home/sheel/birdsong_analysis/src/validation_results'
NUM_SAMPLES_PER_GROUP = 100

# Import Rust Granular Concatenative Synthesizer
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'technical_architecture',
        '/mnt/c/Users/sheel/Desktop/src/technical_architecture/target/release/libtechnical_architecture.so'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['technical_architecture'] = module
    spec.loader.exec_module(module)
    GranularConcatenativeSynthesizer = module.GranularConcatenativeSynthesizer
    print("✅ Successfully imported GranularConcatenativeSynthesizer from Rust")
except ImportError as e:
    print(f"❌ Failed to import Rust synthesizer: {e}")
    print("Make sure the Rust library is compiled with:")
    print("  cargo build --features python-bindings --release")
    sys.exit(1)


def extract_micro_dynamics_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract micro-dynamics features from audio for t-SNE comparison."""
    features = []

    # Time-domain features
    rms = np.sqrt(np.mean(audio ** 2))
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

    # Placeholder for vibrato depth and jitter
    vibrato_depth = 0
    jitter = 0

    # HNR (simplified)
    signal_energy = np.sum(audio ** 2)
    noise_estimate = np.diff(audio)
    noise_energy = np.sum(noise_estimate ** 2)
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
        zcr
    ]

    return np.array(features)


def load_natural_samples(audio_index_path: str, num_samples: int) -> Tuple[np.ndarray, List[str]]:
    """Load natural audio samples from the audio library."""
    print(f"\nLoading natural samples from {audio_index_path}...")

    with open(audio_index_path, 'r') as f:
        audio_index = json.load(f)

    features_list = []
    file_paths = []

    # Sample from phrase types
    phrase_keys = list(audio_index['phrases'].keys())
    random.shuffle(phrase_keys)

    for phrase_key in phrase_keys[:num_samples]:
        phrase_data = audio_index['phrases'][phrase_key]
        segments = phrase_data['segments']

        if not segments:
            continue

        # Load a random segment
        segment = random.choice(segments)

        # Construct file path
        relative_path = segment['relative_path']
        file_path = Path(audio_index_path).parent / relative_path

        if file_path.exists():
            try:
                audio, sr = sf.read(str(file_path))

                # Convert to mono if needed
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                # Extract features
                features = extract_micro_dynamics_features(audio, sr)
                features_list.append(features)
                file_paths.append(str(file_path))
            except Exception:
                continue

        if len(features_list) >= num_samples:
            break

    print(f"  ✅ Loaded {len(features_list)} samples")

    return np.array(features_list), file_paths


def generate_concatenative_samples(audio_index_path: str, num_samples: int) -> Tuple[np.ndarray, List[str]]:
    """Generate concatenative samples by concatenating real audio segments."""
    print("\nGenerating concatenative samples...")

    with open(audio_index_path, 'r') as f:
        audio_index = json.load(f)

    features_list = []
    descriptions = []

    phrase_keys = list(audio_index['phrases'].keys())

    for _ in range(num_samples):
        # Select 2-3 random phrases
        num_phrases = random.randint(2, 3)
        selected_phrases = random.sample(phrase_keys, num_phrases)

        concatenated_audio = []

        for phrase_key in selected_phrases:
            segments = audio_index['phrases'][phrase_key]['segments']
            if segments:
                segment = random.choice(segments)
                relative_path = segment['relative_path']
                file_path = Path(audio_index_path).parent / relative_path

                if file_path.exists():
                    try:
                        audio, sr = sf.read(str(file_path))

                        # Convert to mono if needed
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)

                        concatenated_audio.append(audio)
                    except:
                        continue

        if concatenated_audio:
            # Concatenate
            combined = np.concatenate(concatenated_audio)

            # Extract features from concatenated sample
            features = extract_micro_dynamics_features(combined, 22050)
            features_list.append(features)
            descriptions.append(f"Concatenated_{num_phrases}_phrases")

    print(f"  ✅ Generated {len(features_list)} samples")

    return np.array(features_list), descriptions


def generate_granular_samples(audio_index_path: str, num_samples: int) -> Tuple[np.ndarray, List[str]]:
    """Generate granular synthesis samples using Rust synthesizer."""
    print("\nGenerating Granular Concatenative samples (with pitch/time manipulation)...")

    with open(audio_index_path, 'r') as f:
        audio_index = json.load(f)

    features_list = []
    descriptions = []

    phrase_keys = list(audio_index['phrases'].keys())

    # Create synthesizer
    synthesizer = GranularConcatenativeSynthesizer(sample_rate=22050)

    for i in range(num_samples):
        # Select a random phrase
        phrase_key = random.choice(phrase_keys)
        segments = audio_index['phrases'][phrase_key]['segments']

        if not segments:
            continue

        # Load a random segment as source
        segment = random.choice(segments)
        relative_path = segment['relative_path']
        file_path = Path(audio_index_path).parent / relative_path

        if not file_path.exists():
            continue

        try:
            # Load source audio
            source_audio, sr = sf.read(str(file_path))

            # Convert to mono if needed
            if len(source_audio.shape) > 1:
                source_audio = np.mean(source_audio, axis=1)

            # Resample if needed
            if sr != 22050:
                from scipy import signal
                num_samples = int(len(source_audio) * 22050 / sr)
                source_audio = signal.resample(source_audio, num_samples)
                sr = 22050

            # Load source into synthesizer
            synthesizer.load_source(source_audio.tolist())

            # Apply random pitch shift (0.8 to 1.2 = ±200 cents)
            pitch_shift = random.uniform(0.8, 1.2)
            synthesizer.set_pitch_shift(pitch_shift)

            # Set grain size (15-25ms)
            grain_size = random.uniform(15.0, 25.0)
            synthesizer.set_grain_size_ms(grain_size)

            # Synthesize 100ms audio
            duration_ms = 100.0
            audio_list = synthesizer.synthesize(duration_ms)
            audio = np.array(audio_list, dtype=np.float32)

            # Extract features
            features = extract_micro_dynamics_features(audio, sr)
            features_list.append(features)
            descriptions.append(f"Granular_{phrase_key}_pitch_{pitch_shift:.2f}")

        except Exception as e:
            print(f"    ⚠️  Error processing {phrase_key}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"    Synthesized {i + 1}/{num_samples} samples...")

    print(f"  ✅ Generated {len(features_list)} samples")

    return np.array(features_list), descriptions


def run_tsne_validation(features_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Run t-SNE dimensionality reduction on features."""
    print("\n" + "=" * 80)
    print("RUNNING T-SNE DIMENSIONALITY REDUCTION")
    print("=" * 80)

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
    color_map = {
        'Natural': 'green',
        'Concatenative': 'blue',
        'Granular': 'red'
    }

    marker_map = {
        'Natural': 'o',
        'Concatenative': 's',
        'Granular': '^'
    }

    plt.figure(figsize=(12, 9))

    for group in np.unique(labels):
        mask = labels == group
        plt.scatter(
            projections[mask, 0],
            projections[mask, 1],
            c=color_map.get(group, 'gray'),
            marker=marker_map.get(group, 'o'),
            label=group,
            alpha=0.6,
            s=80
        )

    plt.legend(fontsize=14, loc='best')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Validation: Natural vs Concatenative vs Granular Synthesis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"  ✅ Saved plot to {output_path}")


def calculate_congruence_metrics(projections: np.ndarray, labels: np.ndarray) -> Dict:
    """Calculate statistical metrics of congruence."""
    print("\n" + "=" * 80)
    print("CALCULATING CONGRUENCE METRICS")
    print("=" * 80)

    metrics = {}

    # Get projections for each group
    natural_idx = labels == 'Natural'
    concat_idx = labels == 'Concatenative'
    granular_idx = labels == 'Granular'

    natural_proj = projections[natural_idx]
    concat_proj = projections[concat_idx]
    granular_proj = projections[granular_idx]

    # Calculate centroid distances
    natural_centroid = np.mean(natural_proj, axis=0)
    concat_centroid = np.mean(concat_proj, axis=0)
    granular_centroid = np.mean(granular_proj, axis=0)

    # Distances between centroids
    natural_concat_dist = np.linalg.norm(natural_centroid - concat_centroid)
    natural_granular_dist = np.linalg.norm(natural_centroid - granular_centroid)
    concat_granular_dist = np.linalg.norm(concat_centroid - granular_centroid)

    metrics['centroid_distances'] = {
        'natural_concatenative': float(natural_concat_dist),
        'natural_granular': float(natural_granular_dist),
        'concatenative_granular': float(concat_granular_dist)
    }

    # Calculate spread (standard deviation from centroid)
    natural_spread = np.mean(np.linalg.norm(natural_proj - natural_centroid, axis=1))
    concat_spread = np.mean(np.linalg.norm(concat_proj - concat_centroid, axis=1))
    granular_spread = np.mean(np.linalg.norm(granular_proj - granular_centroid, axis=1))

    metrics['spread'] = {
        'natural': float(natural_spread),
        'concatenative': float(concat_spread),
        'granular': float(granular_spread)
    }

    # Congruence score: lower distance from natural = higher congruence
    # Normalize by natural spread
    granular_congruence = 1.0 / (1.0 + natural_granular_dist / natural_spread)
    metrics['granular_congruence_score'] = float(granular_congruence)

    print("\n  Centroid Distances:")
    print(f"    Natural ↔ Concatenative: {natural_concat_dist:.3f}")
    print(f"    Natural ↔ Granular:      {natural_granular_dist:.3f}")
    print(f"    Concatenative ↔ Granular: {concat_granular_dist:.3f}")

    print("\n  Spread (within-group variance):")
    print(f"    Natural:      {natural_spread:.3f}")
    print(f"    Concatenative: {concat_spread:.3f}")
    print(f"    Granular:     {granular_spread:.3f}")

    print(f"\n  Granular Congruence Score: {granular_congruence:.3f}")
    print("    (1.0 = perfect congruence with natural, 0.0 = no congruence)")

    # Scientific interpretation
    print("\n  " + "=" * 60)
    print("  SCIENTIFIC INTERPRETATION")
    print("=" * 60)

    if natural_granular_dist < 7.0:
        interpretation = "✅ EXCELLENT: Granular synthesis preserves formant structure"
        print(f"  ✅ Natural ↔ Granular distance ({natural_granular_dist:.3f}) < 7.0")
        print("  ✅ Hypothesis CONFIRMED: Granular synthesis achieves high fidelity")
        print("     by preserving spectral envelope (formants) from real audio.")
    elif natural_granular_dist < 15.0:
        interpretation = "⚠️  MODERATE: Granular synthesis shows partial congruence"
        print(f"  ⚠️  Natural ↔ Granular distance ({natural_granular_dist:.3f}) is moderate")
        print("  ⚠️  Granular synthesis is better than additive but not ideal")
    else:
        interpretation = "❌ POOR: Granular synthesis does not preserve formants"
        print(f"  ❌ Natural ↔ Granular distance ({natural_granular_dist:.3f}) is too large")
        print("  ❌ Hypothesis REJECTED: Granular synthesis failed to preserve formants")

    # Comparison with additive synthesis
    print("\n  Comparison with Additive Synthesis (previous result):")
    print("    Additive:     Natural ↔ Dynamic distance ≈ 27.0 (POOR)")
    print(f"    Granular:     Natural ↔ Granular distance ≈ {natural_granular_dist:.1f} ({'GOOD' if natural_granular_dist < 7.0 else 'MODERATE'})")
    print(f"    Improvement:  {(27.0 - natural_granular_dist) / 27.0 * 100:.1f}% reduction in distance")

    metrics['interpretation'] = interpretation

    return metrics


def main():
    """Main validation function."""
    print("=" * 80)
    print("GRANULAR CONCATENATIVE SYNTHESIS VALIDATION")
    print("=" * 80)
    print("\nHypothesis: Granular synthesis achieves t-SNE distance < 7.0")
    print("by preserving formant structure from real audio recordings.")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load samples
    features_dict = {}

    # Natural samples
    natural_features, _ = load_natural_samples(MARMOSET_AUDIO_INDEX, NUM_SAMPLES_PER_GROUP)
    features_dict['Natural'] = natural_features

    # Concatenative samples
    concat_features, _ = generate_concatenative_samples(MARMOSET_AUDIO_INDEX, NUM_SAMPLES_PER_GROUP)
    features_dict['Concatenative'] = concat_features

    # Granular samples
    granular_features, _ = generate_granular_samples(MARMOSET_AUDIO_INDEX, NUM_SAMPLES_PER_GROUP)
    features_dict['Granular'] = granular_features

    # Run t-SNE
    projections, labels = run_tsne_validation(features_dict)

    # Plot results
    plot_path = Path(OUTPUT_DIR) / 'granular_synthesis_tsne_validation.png'
    plot_tsne_results(projections, labels, str(plot_path))

    # Calculate metrics
    metrics = calculate_congruence_metrics(projections, labels)

    # Save results
    results_path = Path(OUTPUT_DIR) / 'granular_synthesis_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Results saved to {results_path}")

    print("\n" + "=" * 80)
    print("✅ GRANULAR SYNTHESIS VALIDATION COMPLETE!")
    print("=" * 80)

    print("\n📂 Output files:")
    print(f"   - Plot: {plot_path}")
    print(f"   - Results: {results_path}")

    print("\n📊 Summary:")
    print(f"   Natural ↔ Concatenative distance: {metrics['centroid_distances']['natural_concatenative']:.3f}")
    print(f"   Natural ↔ Granular distance:      {metrics['centroid_distances']['natural_granular']:.3f}")
    print("   Target: < 7.0")
    print(f"   Result: {'✅ PASS' if metrics['centroid_distances']['natural_granular'] < 7.0 else '❌ FAIL'}")

    print("\n🎯 NEXT STEPS:")
    if metrics['centroid_distances']['natural_granular'] < 7.0:
        print("   1. ✅ Granular synthesis validated!")
        print("   2. Proceed to Bio-Acoustic Turing Test")
        print("   3. Test granular vs natural vocalizations with live animals")
    else:
        print("   1. ⚠️  Granular synthesis needs tuning")
        print("   2. Adjust grain size, window function")
        print("   3. Consider using formant-preserving pitch shifting")

    print("=" * 80)


if __name__ == "__main__":
    main()
