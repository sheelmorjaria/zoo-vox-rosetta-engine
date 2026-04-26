#!/usr/bin/env python3
"""
Spectrogram Comparator for NBD Segments and N-grams
====================================================

This module provides visualization tools for validating the NBD (Neural Boundary
Detection) segmentation pipeline and cluster quality through spectrogram analysis.

Key Visualizations:
1. Segment Spectrogram Grid - Compare multiple segments from same cluster
2. N-gram Storyboard - Visualize syntactic sequences as "comic strips"
3. Boundary Line Plot - Debug NBD boundary detection on raw audio
4. Context Dialect Comparison - Compare territorial vs social patterns

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Audio processing (with fallback)
try:
    import librosa
    import librosa.display
    import soundfile as sf

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: librosa/soundfile not available. Audio visualization disabled.")

# Plotting
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")


def load_cache_segments(cache_dirs: list[str]) -> dict[str, list[dict]]:
    """Load all segments grouped by source file."""
    file_segments: dict[str, list[dict]] = defaultdict(list)

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            continue

        for cache_file in sorted(cache_path.glob("*.json")):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            src = entry.get("source_file", "")
                            if src:
                                file_segments[src].append(entry)
            except Exception:
                pass

    # Sort segments by segment_idx within each file
    for src in file_segments:
        file_segments[src].sort(key=lambda x: x.get("segment_idx", 0))

    return file_segments


def quantize_features(features: list[float], k: int = 1020) -> int:
    """Compute cluster ID from feature vector."""
    if len(features) < 14:
        return 0

    try:
        f0 = int(features[0] * 100.0)
        dur = int(features[1] * 10.0)
        hnr = int(features[6]) if len(features) > 6 else 0
        mfcc1 = int(features[13] * 5.0) if len(features) > 13 else 0

        hash_val = abs(f0 * 1000 + dur * 100 + abs(hnr) * 10 + abs(mfcc1))
        return hash_val % k
    except (TypeError, ValueError):
        return 0


def compute_cluster_assignments(
    file_segments: dict[str, list[dict]], k: int = 1020
) -> dict[str, list[tuple[int, int]]]:
    """Compute cluster IDs for all segments."""
    file_clusters: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)

    for src, segments in file_segments.items():
        for seg in segments:
            features = seg.get("features", [])
            seg_idx = seg.get("segment_idx", 0)
            if features:
                cluster_id = quantize_features(features, k)
                file_clusters[src].append((seg_idx, cluster_id, seg))

    # Sort by segment_idx
    for src in file_clusters:
        file_clusters[src].sort(key=lambda x: x[0])

    return file_clusters


def load_audio_segment(audio_path: str, start_ms: float, end_ms: float) -> tuple[np.ndarray, int]:
    """
    Load a specific time segment from an audio file.

    Returns:
        (audio_data, sample_rate)
    """
    if not AUDIO_AVAILABLE:
        return np.array([]), 0

    try:
        y, sr = sf.read(audio_path)

        # Handle stereo by taking mean
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        # Convert ms to samples
        start_sample = int(start_ms / 1000.0 * sr)
        end_sample = int(end_ms / 1000.0 * sr)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        return y[start_sample:end_sample], sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return np.array([]), 0


def plot_segment_spectrogram(
    audio_data: np.ndarray,
    sr: int,
    ax: plt.Axes,
    title: str = "",
    fmax: int = 100000,  # Bats can go up to 100kHz
    n_mels: int = 128,
) -> Any:
    """Plot a mel spectrogram on the given axes."""
    if len(audio_data) == 0:
        ax.text(0.5, 0.5, "No audio data", ha="center", va="center", transform=ax.transAxes)
        return None

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio_data.astype(np.float32), sr=sr, n_mels=n_mels, fmax=fmax
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=ax)

    ax.set_title(title, fontsize=9)
    ax.label_outer()

    return img


def generate_segment_comparison(
    file_segments: dict[str, list[dict]],
    audio_base_path: str,
    target_cluster_id: int,
    k: int,
    output_path: str,
    max_samples: int = 5,
) -> bool:
    """
    Generate a grid of spectrograms comparing segments from the same cluster.

    This validates cluster purity - segments with the same cluster ID should
    look visually similar.
    """
    if not AUDIO_AVAILABLE or not PLOTTING_AVAILABLE:
        print("Audio/plotting libraries not available")
        return False

    # Find segments with the target cluster ID
    samples = []

    for src, segments in file_segments.items():
        for seg in segments:
            features = seg.get("features", [])
            if not features:
                continue

            cluster_id = quantize_features(features, k)
            if cluster_id == target_cluster_id:
                samples.append((src, seg))

        if len(samples) >= max_samples:
            break

    if not samples:
        print(f"No segments found with cluster ID {target_cluster_id}")
        return False

    # Create plot
    n_samples = min(len(samples), max_samples)
    fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(10, 3 * n_samples))
    fig.suptitle(f"Cluster Purity Check - Cluster {target_cluster_id}", fontsize=14)

    if n_samples == 1:
        axes = [axes]

    for i, (src, seg) in enumerate(samples[:n_samples]):
        # Load audio segment
        audio_path = Path(audio_base_path) / src
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)

        audio_data, sr = load_audio_segment(str(audio_path), start_ms, end_ms)

        # Extract features for title
        features = seg.get("features", [])
        f0 = features[0] if features else 0
        dur = features[1] if len(features) > 1 else 0

        title = f"Sample {i + 1}: F0={f0:.0f}Hz, Dur={dur:.1f}ms | {src}"

        img = plot_segment_spectrogram(audio_data, sr, axes[i], title)

    if img is not None:
        plt.colorbar(img, ax=axes, format="%+2.0f dB", orientation="vertical", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved segment comparison to {output_path}")
    return True


def generate_ngram_storyboard(
    file_segments: dict[str, list[dict]],
    audio_base_path: str,
    ngram_ids: list[int],
    k: int,
    output_path: str,
) -> bool:
    """
    Generate a "storyboard" visualization for an N-gram sequence.

    This shows how the spectrogram evolves across the sequence,
    validating that the N-gram represents a coherent acoustic phrase.
    """
    if not AUDIO_AVAILABLE or not PLOTTING_AVAILABLE:
        print("Audio/plotting libraries not available")
        return False

    # Find a file containing this N-gram sequence
    ngram_len = len(ngram_ids)
    found_sequence = None

    for src, segments in file_segments.items():
        # Compute cluster IDs for this file
        cluster_ids = []
        for seg in segments:
            features = seg.get("features", [])
            cluster_ids.append(quantize_features(features, k))

        # Search for N-gram
        for i in range(len(cluster_ids) - ngram_len + 1):
            if cluster_ids[i : i + ngram_len] == ngram_ids:
                found_sequence = (src, segments[i : i + ngram_len])
                break

        if found_sequence:
            break

    if not found_sequence:
        print(f"N-gram sequence {ngram_ids} not found in corpus")
        return False

    src, segs = found_sequence

    # Create storyboard
    fig, axes = plt.subplots(nrows=1, ncols=ngram_len, figsize=(5 * ngram_len, 4))

    if ngram_len == 1:
        axes = [axes]

    fig.suptitle(f"N-Gram Storyboard: {ngram_ids}", fontsize=14)

    for i, seg in enumerate(segs):
        # Load audio segment
        audio_path = Path(audio_base_path) / src
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)

        audio_data, sr = load_audio_segment(str(audio_path), start_ms, end_ms)

        # Extract features
        features = seg.get("features", [])
        f0 = features[0] if features else 0
        dur = features[1] if len(features) > 1 else 0

        title = f"Unit {i + 1}: ID {ngram_ids[i]}\nF0={f0:.0f}Hz, Dur={dur:.1f}ms"

        img = plot_segment_spectrogram(audio_data, sr, axes[i], title)

        # Add red border to indicate segment boundary
        for spine in axes[i].spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(2)

    if img is not None:
        plt.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved N-gram storyboard to {output_path}")
    return True


def generate_boundary_plot(
    file_segments: dict[str, list[dict]],
    audio_base_path: str,
    source_file: str,
    output_path: str,
    max_segments: int = 10,
) -> bool:
    """
    Plot raw audio waveform with NBD boundary markers.

    This validates that the NBD boundaries are correctly detecting
    semantic boundaries in the audio (not cutting off calls or
    including excessive silence).
    """
    if not AUDIO_AVAILABLE or not PLOTTING_AVAILABLE:
        print("Audio/plotting libraries not available")
        return False

    if source_file not in file_segments:
        print(f"File {source_file} not found in cache")
        return False

    segments = file_segments[source_file][:max_segments]

    # Load full audio
    audio_path = Path(audio_base_path) / source_file
    try:
        y, sr = sf.read(str(audio_path))
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return False

    # Create plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

    # Top: Waveform with boundaries
    librosa.display.waveshow(y.astype(np.float32), sr=sr, ax=axes[0], alpha=0.6)

    for seg in segments:
        start_s = seg.get("start_ms", 0) / 1000.0
        end_s = seg.get("end_ms", 0) / 1000.0
        _boundary_type = seg.get("boundary_type", "Unknown")  # noqa: F841

        # Draw boundary lines
        axes[0].axvline(x=start_s, color="green", linestyle="--", alpha=0.8, linewidth=1.5)
        axes[0].axvline(x=end_s, color="red", linestyle="--", alpha=0.8, linewidth=1.5)

        # Label segment
        mid = (start_s + end_s) / 2
        axes[0].text(
            mid,
            0.8,
            f"S{seg.get('segment_idx', 0)}",
            ha="center",
            fontsize=8,
            color="blue",
            transform=axes[0].get_xaxis_transform(),
        )

    axes[0].set_title(f"NBD Boundary Detection - {source_file}")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(["Waveform", "Start", "End"])

    # Bottom: Full spectrogram
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_mels=128, fmax=100000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=100000, ax=axes[1]
    )

    # Overlay boundary lines on spectrogram
    for seg in segments:
        start_s = seg.get("start_ms", 0) / 1000.0
        end_s = seg.get("end_ms", 0) / 1000.0

        axes[1].axvline(x=start_s, color="green", linestyle="--", alpha=0.8, linewidth=1.5)
        axes[1].axvline(x=end_s, color="red", linestyle="--", alpha=0.8, linewidth=1.5)

    axes[1].set_title("Spectrogram with Segment Boundaries")

    plt.colorbar(img, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved boundary plot to {output_path}")
    return True


def generate_context_dialect_comparison(
    file_segments: dict[str, list[dict]],
    audio_base_path: str,
    k: int,
    output_path: str,
) -> bool:
    """
    Compare spectrograms across different behavioral contexts.

    This validates the "Context Dialect" hypothesis - that different
    contexts (territorial, social, etc.) have distinct acoustic patterns.
    """
    if not AUDIO_AVAILABLE or not PLOTTING_AVAILABLE:
        print("Audio/plotting libraries not available")
        return False

    # Group segments by context
    context_samples: dict[int, list[tuple[str, dict]]] = defaultdict(list)

    for src, segments in file_segments.items():
        for seg in segments:
            context = seg.get("context", 0)
            if len(context_samples[context]) < 3:  # 3 samples per context
                context_samples[context].append((src, seg))

    # Get top contexts
    top_contexts = sorted(
        context_samples.keys(), key=lambda c: len(context_samples[c]), reverse=True
    )[:4]

    if not top_contexts:
        print("No context data found")
        return False

    # Create comparison plot
    n_contexts = len(top_contexts)
    fig, axes = plt.subplots(nrows=n_contexts, ncols=3, figsize=(15, 4 * n_contexts))

    if n_contexts == 1:
        axes = axes.reshape(1, -1)

    context_names = {
        0: "Unknown",
        1: "Food-related",
        2: "Social",
        3: "Territorial",
        4: "Aggression",
        5: "Mating",
        6: "Distress",
        7: "Exploration",
        8: "Sleep",
        9: "Grooming",
        10: "Mother-Infant",
        11: "Territorial",
        12: "Social",
    }

    for row, context in enumerate(top_contexts):
        samples = context_samples[context][:3]
        context_name = context_names.get(context, f"Context {context}")

        for col, (src, seg) in enumerate(samples):
            audio_path = Path(audio_base_path) / src
            start_ms = seg.get("start_ms", 0)
            end_ms = seg.get("end_ms", 0)

            audio_data, sr = load_audio_segment(str(audio_path), start_ms, end_ms)

            features = seg.get("features", [])
            f0 = features[0] if features else 0
            _dur = features[1] if len(features) > 1 else 0  # noqa: F841

            title = f"{context_name}\nF0={f0:.0f}Hz"

            plot_segment_spectrogram(audio_data, sr, axes[row, col], title)

        # Hide empty columns
        for col in range(len(samples), 3):
            axes[row, col].set_visible(False)

    fig.suptitle("Context Dialect Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved context dialect comparison to {output_path}")
    return True


def generate_full_sequence_spectrogram(
    file_segments: dict[str, list[dict]],
    audio_base_path: str,
    source_file: str,
    output_path: str,
    k: int = 1020,
) -> bool:
    """
    Generate a complete spectrogram for a vocalization with segment annotations.
    """
    if not AUDIO_AVAILABLE or not PLOTTING_AVAILABLE:
        print("Audio/plotting libraries not available")
        return False

    if source_file not in file_segments:
        print(f"File {source_file} not found in cache")
        return False

    segments = file_segments[source_file]

    # Load full audio
    audio_path = Path(audio_base_path) / source_file
    try:
        y, sr = sf.read(str(audio_path))
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return False

    # Compute cluster IDs
    cluster_ids = [quantize_features(seg.get("features", []), k) for seg in segments]

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Spectrogram
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_mels=128, fmax=100000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=100000, ax=ax)

    # Add segment annotations
    for i, seg in enumerate(segments):
        start_s = seg.get("start_ms", 0) / 1000.0
        end_s = seg.get("end_ms", 0) / 1000.0
        mid = (start_s + end_s) / 2

        # Draw boundary lines
        ax.axvline(x=start_s, color="white", linestyle="-", alpha=0.5, linewidth=1)
        ax.axvline(x=end_s, color="white", linestyle="-", alpha=0.5, linewidth=1)

        # Add cluster ID label
        ax.text(
            mid,
            90000,
            f"{cluster_ids[i]}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="yellow",
            fontweight="bold",
        )

    ax.set_title(f"Full Sequence: {source_file}\nCluster IDs: {cluster_ids}")

    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved full sequence spectrogram to {output_path}")
    return True


def main():
    print("=" * 80)
    print("SPECTROGRAM COMPARATOR FOR NBD SEGMENTS AND N-GRAMS")
    print("=" * 80)

    if not AUDIO_AVAILABLE:
        print("\nERROR: librosa and soundfile are required for audio visualization.")
        print("Install with: pip install librosa soundfile")
        return

    if not PLOTTING_AVAILABLE:
        print("\nERROR: matplotlib is required for plotting.")
        print("Install with: pip install matplotlib")
        return

    # Configuration
    cache_dirs = [
        "bat_nbd_cache_parallel",
        "bat_fm_cache",
        "bat_nbd_cache_full",
    ]
    audio_base_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio"
    output_dir = Path("spectrogram_analysis")
    output_dir.mkdir(exist_ok=True)

    k = 1020  # Vocabulary size

    print("\n[1] Loading cache data...")
    file_segments = load_cache_segments(cache_dirs)
    print(f"Loaded {len(file_segments)} files")

    # Get a sample file for boundary analysis
    sample_files = list(file_segments.keys())[:3]
    print(f"Sample files: {sample_files}")

    # Generate visualizations
    print("\n[2] Generating boundary detection plots...")
    for i, src in enumerate(sample_files):
        output_path = output_dir / f"boundary_check_{src.replace('.wav', '')}.png"
        generate_boundary_plot(file_segments, audio_base_path, src, str(output_path))

    print("\n[3] Generating segment comparison (cluster purity check)...")
    # Use a few different cluster IDs to check purity
    for cluster_id in [42, 100, 500, 764]:
        output_path = output_dir / f"cluster_{cluster_id}_comparison.png"
        generate_segment_comparison(file_segments, audio_base_path, cluster_id, k, str(output_path))

    print("\n[4] Generating N-gram storyboards...")
    # Top bigrams from corpus analysis
    ngrams = [
        [764, 304],
        [384, 464],
        [114, 464, 604],  # LRN-6 prefix
    ]

    for ngram in ngrams:
        output_path = output_dir / f"ngram_{'_'.join(map(str, ngram))}_storyboard.png"
        generate_ngram_storyboard(file_segments, audio_base_path, ngram, k, str(output_path))

    print("\n[5] Generating context dialect comparison...")
    output_path = output_dir / "context_dialect_comparison.png"
    generate_context_dialect_comparison(file_segments, audio_base_path, k, str(output_path))

    print("\n[6] Generating full sequence spectrograms...")
    for src in sample_files[:2]:
        output_path = output_dir / f"full_sequence_{src.replace('.wav', '')}.png"
        generate_full_sequence_spectrogram(file_segments, audio_base_path, src, str(output_path), k)

    print("\n" + "=" * 80)
    print("VISUALIZATION VALIDATION CHECKLIST")
    print("=" * 80)
    print("""
Boundary Quality (NBD Validation):
  [ ] The "Cut-off" Check: Does the spectrogram start abruptly?
  [ ] The "Slack" Check: Is there excessive silence at start/end?
  [ ] Ideal: Spectrogram starts at energy onset, ends as energy fades.

Cluster Purity (Clustering Validation):
  [ ] Visual Similarity: Do segments from same cluster look alike?
  [ ] The "Intruder": Are there mismatched patterns in the same cluster?

Feature-to-Image Correlation:
  [ ] F0 matches frequency content on spectrogram
  [ ] Duration matches plot width
  [ ] Energy matches brightness

N-Gram Coherence:
  [ ] Texture Flow: Spectrograms flow smoothly across panels
  [ ] Energy Envelope: Amplitude creates phrase shape
  [ ] Pitch Contour: Frequency lines connect across segments
    """)

    print(f"\nAll visualizations saved to: {output_dir}")
    print("Spectrogram Comparator Complete.")


if __name__ == "__main__":
    main()
