#!/usr/bin/env python3
"""
Test Adaptive Gap Threshold Enhancement

Tests the new adaptive gap threshold feature for TRANSIENT/RHYTHMIC modalities.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone, Modality

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def create_synthetic_click_train(sr, duration_sec, click_interval_ms=10):
    """Create synthetic click train for testing."""
    num_samples = int(duration_sec * sr)
    audio = np.zeros(num_samples)

    # Add clicks at regular intervals
    interval_samples = int(click_interval_ms * sr / 1000)
    for i in range(0, num_samples, interval_samples):
        if i + 100 < num_samples:
            # Add click (short pulse)
            audio[i:i+100] = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(100) / sr)

    return audio


def create_synthetic_harmonic_tone(sr, duration_sec, frequency_hz=440):
    """Create synthetic harmonic tone for comparison."""
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples)
    return 0.5 * np.sin(2 * np.pi * frequency_hz * t)


def test_adaptive_gap_detection():
    """Test adaptive gap threshold with synthetic signals."""
    print("="*70)
    print("ADAPTIVE GAP THRESHOLD TEST")
    print("="*70)

    sr = 48000

    # Test 1: Dense click train (TRANSIENT)
    print("\n📊 Test 1: Dense Click Train (TRANSIENT)")
    print("-" * 70)

    click_train = create_synthetic_click_train(sr, 1.0, click_interval_ms=10)
    analyzer = UniversalRosettaStone(sample_rate=sr)

    # Check overall modality detection
    modality = analyzer._detect_overall_modality(click_train)
    print(f"  Overall modality: {modality.name}")

    # Check adaptive gap calculation
    adaptive_gap = analyzer._calculate_adaptive_gap_threshold(click_train)
    print(f"  Adaptive gap threshold: {adaptive_gap:.2f} ms")

    # Test segmentation with adaptive gap
    phrases_adaptive = analyzer.segment_phrases(click_train, use_adaptive_gap=True)
    print(f"  Phrases detected (adaptive): {len(phrases_adaptive)}")

    # Test segmentation without adaptive gap
    phrases_fixed = analyzer.segment_phrases(click_train, min_gap_ms=50.0, use_adaptive_gap=False)
    print(f"  Phrases detected (fixed 50ms): {len(phrases_fixed)}")

    # Test 2: Harmonic tone (should NOT use adaptive gap)
    print("\n📊 Test 2: Harmonic Tone (HARMONIC)")
    print("-" * 70)

    harmonic_tone = create_synthetic_harmonic_tone(sr, 1.0, frequency_hz=440)
    analyzer2 = UniversalRosettaStone(sample_rate=sr)

    modality = analyzer2._detect_overall_modality(harmonic_tone)
    print(f"  Overall modality: {modality.name}")

    adaptive_gap = analyzer2._calculate_adaptive_gap_threshold(harmonic_tone)
    print(f"  Adaptive gap threshold: {adaptive_gap:.2f} ms")

    # Test segmentation (should not use adaptive gap for HARMONIC)
    phrases_adaptive = analyzer2.segment_phrases(harmonic_tone, use_adaptive_gap=True)
    print(f"  Phrases detected: {len(phrases_adaptive)}")

    print("\n" + "="*70)
    print("✅ Synthetic tests complete!")
    print("="*70)


def test_real_sperm_whale_audio():
    """Test with real sperm whale audio if available."""
    if not HAS_SOUNDFILE:
        return

    base_dir = Path.home() / "birdsong_analysis/data/Dominica_dataset/Signal_parts"
    test_file = base_dir / "SW_19_filtered.wav"

    if not test_file.exists():
        print("\n⚠️  Sperm whale test file not found, skipping real audio test")
        return

    print("\n" + "="*70)
    print("REAL SPERM WHALE AUDIO TEST")
    print("="*70)

    audio, sr = sf.read(test_file)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Use first 10 seconds for quick test
    audio = audio[:10 * sr]

    print(f"\n📊 File: {test_file.name}")
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Sample rate: {sr} Hz")

    analyzer = UniversalRosettaStone(sample_rate=sr)

    # Detect modality
    modality = analyzer._detect_overall_modality(audio)
    print(f"  Overall modality: {modality.name}")

    # Calculate adaptive gap
    adaptive_gap = analyzer._calculate_adaptive_gap_threshold(audio)
    print(f"  Adaptive gap threshold: {adaptive_gap:.2f} ms")

    # Test with adaptive gap
    print("\n  Segmentation comparison:")
    phrases_adaptive = analyzer.segment_phrases(
        audio,
        min_gap_ms=50.0,
        use_adaptive_gap=True
    )
    print(f"    With adaptive gap (max 50ms): {len(phrases_adaptive)} phrases")

    # Test without adaptive gap
    phrases_fixed_50 = analyzer.segment_phrases(
        audio,
        min_gap_ms=50.0,
        use_adaptive_gap=False
    )
    print(f"    With fixed 50ms gap: {len(phrases_fixed_50)} phrases")

    # Test with larger fixed gap
    phrases_fixed_100 = analyzer.segment_phrases(
        audio,
        min_gap_ms=100.0,
        use_adaptive_gap=False
    )
    print(f"    With fixed 100ms gap: {len(phrases_fixed_100)} phrases")

    # Show improvement
    if len(phrases_adaptive) > 0:
        print(f"\n  ✅ Adaptive gap detection working!")
    else:
        print(f"\n  ⚠️  No phrases detected - may need parameter adjustment")

    print("="*70)


def main():
    """Run all tests."""
    test_adaptive_gap_detection()
    test_real_sperm_whale_audio()


if __name__ == "__main__":
    main()
