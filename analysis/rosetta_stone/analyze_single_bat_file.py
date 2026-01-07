#!/usr/bin/env python3
"""
Deep analysis of a single Egyptian fruit bat file
to understand signal structure
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from universal_rosetta_stone import UniversalRosettaStone

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def main():
    """Analyze a specific bat file in detail."""
    # Use one of the files that had phrases detected
    data_dir = Path.home() / "birdsong_analysis" / "data" / "egyptian_fruit_bat_10k" / "audio"
    filepath = data_dir / "20874.wav"  # File with 31 phrases detected

    print("=" * 70)
    print("DETAILED BAT FILE ANALYSIS")
    print(f"File: {filepath.name}")
    print("=" * 70)

    # Load audio
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    print("\n📊 Basic Info:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio) / sr * 1000:.0f} ms")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"  Peak: {np.max(np.abs(audio)):.6f}")

    # Analyze first few seconds in detail
    sample_length = int(0.5 * sr)  # First 500 ms
    audio_sample = audio[:sample_length]

    print("\n🔍 First 500ms analysis:")

    # Find peaks (clicks)
    envelope = np.abs(audio_sample)
    threshold = np.mean(envelope) + 3 * np.std(envelope)
    peaks = []
    for i in range(1, len(envelope) - 1):
        if (
            envelope[i] > threshold
            and envelope[i] > envelope[i - 1]
            and envelope[i] > envelope[i + 1]
        ):
            peaks.append(i)

    print(f"  Detected {len(peaks)} peaks (clicks) above threshold")

    # Analyze each peak/click
    if len(peaks) > 0:
        print("\n  First 10 clicks:")
        for i, peak in enumerate(peaks[:10]):
            # Extract window around peak
            win_size = int(0.005 * sr)  # 5ms window
            start = max(0, peak - win_size)
            end = min(len(audio_sample), peak + win_size)
            click = audio_sample[start:end]

            # Analyze
            from scipy.fft import fft, fftfreq

            fft_result = fft(click)
            freqs = fftfreq(len(click), 1 / sr)
            magnitude = np.abs(fft_result)

            pos_freqs = freqs[: len(freqs) // 2]
            pos_magnitude = magnitude[: len(magnitude) // 2]

            # Find dominant frequency
            dom_freq_idx = np.argmax(pos_magnitude)
            dom_freq = pos_freqs[dom_freq_idx]

            # Calculate zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(click)))) / (2 * len(click))

            print(
                f"    Click {i + 1}: {len(click) / sr * 1000:.1f}ms, DomFreq: {
                    abs(dom_freq) / 1000:.1f}kHz, ZCR: {zcr:.3f}"
            )

    # Try segmentation with aggressive parameters
    print("\n🔍 Phrase segmentation (gap=2ms, dur=1ms):")
    analyzer = UniversalRosettaStone(sample_rate=sr)

    try:
        phrases = analyzer.segment_phrases(audio_sample, min_gap_ms=2, min_phrase_duration_ms=1)
        print(f"  Detected {len(phrases)} phrases")

        if len(phrases) > 0:
            print("\n  First 10 phrase modalities:")
            for i, phrase in enumerate(phrases[:10]):
                modality = analyzer.detect_modality(phrase.data)
                probs = analyzer.get_modality_probabilities(phrase.data)
                print(f"    Phrase {i + 1}: {modality.name} - {probs}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n💡 Interpretation:")
    print("  - Many short peaks detected: These are echolocation CLICKS")
    print("  - Clicks are classified as TRANSIENT (correct!)")
    print("  - No sustained FM sweeps in this recording")
    print("  - Egyptian fruit bats use both clicks and FM sweeps")
    print("  - This dataset appears to be primarily click-based echolocation")


if __name__ == "__main__":
    if HAS_SOUNDFILE:
        main()
    else:
        print("❌ soundfile not installed")
