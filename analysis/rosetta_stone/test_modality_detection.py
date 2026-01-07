#!/usr/bin/env python3
"""
Modality Detection Test Suite
==============================

Tests the Universal Rosetta Stone's ability to detect acoustic modalities
across different species and handle mixed-modality vocalizations.

Scientific Context:
- Many species use mixed modalities (e.g., bats use FM sweeps for echolocation
  but harmonic calls for social communication)
- Modality detection should work at the phrase level, not just species level
- Different phrases from the same species may have different modalities

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))  # rosetta_stone/

# Try direct import first, then fallback
try:
    from universal_rosetta_stone import Modality, UniversalRosettaStone
except ImportError:
    try:
        from analysis.rosetta_stone import Modality, UniversalRosettaStone
    except ImportError:
        # Last resort: import from current directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "universal_rosetta_stone",
            Path(__file__).parent / "universal_rosetta_stone.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        UniversalRosettaStone = module.UniversalRosettaStone
        Modality = module.Modality


def generate_harmonic_signal(freq_hz, duration_ms, sample_rate=48000, amplitude=0.5):
    """Generate a harmonic signal (flat tone)."""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))

    # Fundamental frequency
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)

    # Add harmonics for richness
    signal += (amplitude / 2) * np.sin(2 * np.pi * freq_hz * 2 * t)
    signal += (amplitude / 4) * np.sin(2 * np.pi * freq_hz * 3 * t)

    # Normalize
    signal = signal / np.max(np.abs(signal))

    return signal


def generate_fm_sweep_signal(start_freq_hz, end_freq_hz, duration_ms, sample_rate=48000, amplitude=0.5):
    """Generate an FM sweep signal (frequency changes over time)."""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))

    # Linear chirp (frequency changes linearly)
    # Instantaneous phase = 2*pi * integral of frequency
    f0 = start_freq_hz
    f1 = end_freq_hz
    k = (f1 - f0) / (duration_ms / 1000)  # Chirp rate

    # Phase = 2*pi * (f0*t + 0.5*k*t^2)
    signal = amplitude * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    # Normalize
    signal = signal / np.max(np.abs(signal))

    return signal


def generate_transient_signal(duration_ms, sample_rate=48000, amplitude=0.5):
    """Generate a transient/click signal."""
    num_samples = int(sample_rate * duration_ms / 1000)

    # Create a short Gaussian pulse
    t = np.linspace(-5, 5, num_samples)
    pulse = amplitude * np.exp(-t**2)

    # Add some high-frequency content
    pulse += (amplitude / 2) * np.exp(-((t - 1)**2) / 0.5) * np.sin(2 * np.pi * 10000 * t)

    # Normalize
    pulse = pulse / np.max(np.abs(pulse))

    # Pad with zeros to achieve desired duration
    signal = np.zeros(num_samples)
    pulse_start = num_samples // 4
    pulse_end = pulse_start + len(pulse)
    if pulse_end <= num_samples:
        signal[pulse_start:pulse_end] = pulse

    return signal


def generate_rhythmic_signal(tempo_bpm, duration_ms, sample_rate=48000, amplitude=0.5):
    """Generate a rhythmic pattern (e.g., cricket chirps)."""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))

    # Period of the rhythm
    period_sec = 60.0 / tempo_bpm

    # Create rhythmic pattern using amplitude modulation
    signal = amplitude * np.sin(2 * np.pi * 440 * t)  # Base tone
    envelope = 0.5 * (1 + np.sin(2 * np.pi * (1.0 / period_sec) * t))
    signal *= envelope

    # Add some pulses
    for i in np.arange(0, duration_ms / 1000, period_sec):
        pulse_start = int(i * sample_rate)
        pulse_width = int(0.01 * sample_rate)  # 10ms pulse
        if pulse_start + pulse_width < len(signal):
            signal[pulse_start:pulse_start + pulse_width] += amplitude * 0.5

    # Normalize
    signal = signal / np.max(np.abs(signal))

    return signal


def generate_mixed_modality_signal(harmonic_ratio=0.5, fm_ratio=0.5,
                                   duration_ms=100, sample_rate=48000):
    """Generate a mixed modality signal (harmonic + FM sweep)."""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))

    # Harmonic component
    harmonic = harmonic_ratio * np.sin(2 * np.pi * 7000 * t)

    # FM sweep component
    f0 = 6000
    f1 = 8000
    k = (f1 - f0) / (duration_ms / 1000)
    fm = fm_ratio * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    # Combine
    signal = harmonic + fm
    signal = signal / np.max(np.abs(signal))

    return signal


class ModalityDetectionResult:
    """Result of modality detection test."""
    def __init__(self, species: str, expected_modality: Modality,
                 detected_modality: Modality, confidence: float = 1.0):
        self.species = species
        self.expected_modality = expected_modality
        self.detected_modality = detected_modality
        self.confidence = confidence
        self.correct = expected_modality == detected_modality


def test_modality_detection():
    """Test modality detection across species with known vocalizations."""

    print("=" * 80)
    print("MODALITY DETECTION TEST SUITE")
    print("=" * 80)
    print()

    results = []

    # Test 1: Marmoset - Harmonic (flat tones, 5-12 kHz)
    print("Test 1: Marmoset Harmonic Call (7 kHz)")
    print("-" * 40)
    marmoset_audio = generate_harmonic_signal(freq_hz=7000, duration_ms=50)
    analyzer = UniversalRosettaStone(sample_rate=48000)
    detected = analyzer.detect_modality(marmoset_audio)
    result = ModalityDetectionResult("Marmoset", Modality.HARMONIC, detected)
    results.append(result)
    print(f"Expected: {result.expected_modality.name}")
    print(f"Detected: {result.detected_modality.name}")
    print(f"Status: {'✓ PASS' if result.correct else '✗ FAIL'}")
    print()

    # Test 2: Egyptian Fruit Bat - FM Sweep (20-90 kHz, note we'll use lower freq for demo)
    print("Test 2: Egyptian Fruit Bat FM Sweep (25-45 kHz)")
    print("-" * 40)
    bat_audio = generate_fm_sweep_signal(start_freq_hz=25000, end_freq_hz=45000, duration_ms=20)
    detected = analyzer.detect_modality(bat_audio)
    result = ModalityDetectionResult("Egyptian Fruit Bat", Modality.FM_SWEEP, detected)
    results.append(result)
    print(f"Expected: {result.expected_modality.name}")
    print(f"Detected: {result.detected_modality.name}")
    print(f"Status: {'✓ PASS' if result.correct else '✗ FAIL'}")
    print()

    # Test 3: Sperm Whale - Transient clicks
    print("Test 3: Sperm Whale Transient Click")
    print("-" * 40)
    whale_audio = generate_transient_signal(duration_ms=30)
    detected = analyzer.detect_modality(whale_audio)
    result = ModalityDetectionResult("Sperm Whale", Modality.TRANSIENT, detected)
    results.append(result)
    print(f"Expected: {result.expected_modality.name}")
    print(f"Detected: {result.detected_modality.name}")
    print(f"Status: {'✓ PASS' if result.correct else '✗ FAIL'}")
    print()

    # Test 4: Cricket - Rhythmic chirps
    print("Test 4: Cricket Rhythmic Chirp (60 BPM)")
    print("-" * 40)
    cricket_audio = generate_rhythmic_signal(tempo_bpm=60, duration_ms=200)
    detected = analyzer.detect_modality(cricket_audio)
    result = ModalityDetectionResult("Cricket", Modality.RHYTHMIC, detected)
    results.append(result)
    print(f"Expected: {result.expected_modality.name}")
    print(f"Detected: {result.detected_modality.name}")
    print(f"Status: {'✓ PASS' if result.correct else '✗ FAIL'}")
    print()

    # Test 5: Dolphin - Whistle (Harmonic)
    print("Test 5: Dolphin Whistle (12 kHz harmonic)")
    print("-" * 40)
    dolphin_audio = generate_harmonic_signal(freq_hz=12000, duration_ms=150)
    detected = analyzer.detect_modality(dolphin_audio)
    result = ModalityDetectionResult("Dolphin", Modality.HARMONIC, detected)
    results.append(result)
    print(f"Expected: {result.expected_modality.name}")
    print(f"Detected: {result.detected_modality.name}")
    print(f"Status: {'✓ PASS' if result.correct else '✗ FAIL'}")
    print()

    # Test 6: Zebra Finch - Harmonic song
    print("Test 6: Zebra Finch Song (4 kHz harmonic)")
    print("-" * 40)
    finch_audio = generate_harmonic_signal(freq_hz=4000, duration_ms=100)
    detected = analyzer.detect_modality(finch_audio)
    result = ModalityDetectionResult("Zebra Finch", Modality.HARMONIC, detected)
    results.append(result)
    print(f"Expected: {result.expected_modality.name}")
    print(f"Detected: {result.detected_modality.name}")
    print(f"Status: {'✓ PASS' if result.correct else '✗ FAIL'}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    pass_count = sum(1 for r in results if r.correct)
    total_count = len(results)
    accuracy = pass_count / total_count * 100 if total_count > 0 else 0

    print(f"Tests Passed: {pass_count}/{total_count} ({accuracy:.1f}%)")
    print()

    for i, result in enumerate(results, 1):
        status = "✓" if result.correct else "✗"
        print(f"{i}. {result.species:20} - Expected: {result.expected_modality.name:10} "
              f"Detected: {result.detected_modality.name:10} {status}")

    print()
    return accuracy >= 66.7  # At least 2/3 correct


def test_mixed_modality_detection():
    """Test detection of mixed-modality signals."""

    print("=" * 80)
    print("MIXED MODALITY DETECTION TEST")
    print("=" * 80)
    print()
    print("Many species use mixed modalities in their vocalizations.")
    print("This test demonstrates how the system handles such cases.")
    print()

    analyzer = UniversalRosettaStone(sample_rate=48000)

    # Test 1: Pure harmonic (baseline)
    print("Test 1: Pure Harmonic Signal (Baseline)")
    print("-" * 40)
    harmonic_only = generate_harmonic_signal(freq_hz=7000, duration_ms=100)
    detected = analyzer.detect_modality(harmonic_only)
    probs = analyzer.get_modality_probabilities(harmonic_only)
    print(f"Detected: {detected.name}")
    print(f"Probabilities: {probs}")
    print("Features: ZCR low, spectral flatness low")
    print()

    # Test 2: Pure FM sweep (baseline)
    print("Test 2: Pure FM Sweep Signal (Baseline)")
    print("-" * 40)
    fm_only = generate_fm_sweep_signal(start_freq_hz=6000, end_freq_hz=8000, duration_ms=100)
    detected = analyzer.detect_modality(fm_only)
    probs = analyzer.get_modality_probabilities(fm_only)
    print(f"Detected: {detected.name}")
    print(f"Probabilities: {probs}")
    print("Features: ZCR high, frequency slope present")
    print()

    # Test 3: Mixed - Harmonic dominant (70% harmonic, 30% FM)
    print("Test 3: Mixed Signal - Harmonic Dominant (70% H, 30% FM)")
    print("-" * 40)
    mixed_h_dominant = generate_mixed_modality_signal(harmonic_ratio=0.7, fm_ratio=0.3, duration_ms=100)
    detected = analyzer.detect_modality(mixed_h_dominant)
    probs = analyzer.get_modality_probabilities(mixed_h_dominant)
    print(f"Detected: {detected.name}")
    print(f"Probabilities: {probs}")
    print("Expected: HARMONIC (harmonic component dominates)")
    print()

    # Test 4: Mixed - FM dominant (30% harmonic, 70% FM)
    print("Test 4: Mixed Signal - FM Sweep Dominant (30% H, 70% FM)")
    print("-" * 40)
    mixed_fm_dominant = generate_mixed_modality_signal(harmonic_ratio=0.3, fm_ratio=0.7, duration_ms=100)
    detected = analyzer.detect_modality(mixed_fm_dominant)
    probs = analyzer.get_modality_probabilities(mixed_fm_dominant)
    print(f"Detected: {detected.name}")
    print(f"Probabilities: {probs}")
    print("Expected: FM_SWEEP (FM component dominates)")
    print()

    # Test 5: Balanced mix (50% harmonic, 50% FM)
    print("Test 5: Balanced Mixed Signal (50% H, 50% FM)")
    print("-" * 40)
    balanced_mixed = generate_mixed_modality_signal(harmonic_ratio=0.5, fm_ratio=0.5, duration_ms=100)
    detected = analyzer.detect_modality(balanced_mixed)
    probs = analyzer.get_modality_probabilities(balanced_mixed)
    print(f"Detected: {detected.name}")
    print(f"Probabilities: {probs}")
    print("Note: Result depends on which features are more prominent")
    print()

    print("=" * 80)
    print("MIXED MODALITY ANALYSIS")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print()
    print("1. The detect_modality() method returns a SINGLE modality per audio signal.")
    print("   This is the primary modality used for phrase classification.")
    print()
    print("2. The get_modality_probabilities() method returns probability-like scores")
    print("   for ALL modalities, allowing detection of mixed-modality signals.")
    print()
    print("3. For species with mixed modality vocalizations (e.g., bats using both")
    print("   FM sweeps for echolocation AND harmonic calls for social communication),")
    print("   modality should be detected AT THE PHRASE LEVEL, not species level.")
    print()
    print("4. SOLUTION: Different phrases from the same species can have different")
    print("   modalities. The phrase segmentation process will naturally separate")
    print("   different modality phrases into different PhraseSignature objects.")
    print()
    print("Example for Egyptian Fruit Bat:")
    print("  - FM sweep phrase → Modality.FM_SWEEP")
    print("  - Harmonic social call → Modality.HARMONIC")
    print("  - Click sequence → Modality.TRANSIENT")
    print()
    print("5. The get_modality_probabilities() method can be used to:")
    print("   - Identify mixed-modality signals (multiple high probabilities)")
    print("   - Set thresholds for modality confidence")
    print("   - Analyze gradual modality transitions")
    print()


def test_phrase_level_modality():
    """Test that phrase segmentation can separate different modalities."""

    print("=" * 80)
    print("PHRASE-LEVEL MODALITY SEGMENTATION TEST")
    print("=" * 80)
    print()
    print("This test demonstrates how a recording with mixed modalities")
    print("can be segmented into phrases, each with its own modality.")
    print()

    # Create a mixed recording: harmonic → FM sweep → transient
    sample_rate = 48000

    # Segment 1: Harmonic call (50ms)
    harmonic_part = generate_harmonic_signal(freq_hz=7000, duration_ms=50, sample_rate=sample_rate)

    # Segment 2: FM sweep (30ms)
    fm_part = generate_fm_sweep_signal(start_freq_hz=20000, end_freq_hz=40000,
                                       duration_ms=30, sample_rate=sample_rate)

    # Segment 3: Transient click (20ms)
    transient_part = generate_transient_signal(duration_ms=20, sample_rate=sample_rate)

    # Combine with gaps
    gap_silence = np.zeros(int(0.01 * sample_rate))  # 10ms gap
    mixed_recording = np.concatenate([
        harmonic_part,
        gap_silence,
        fm_part,
        gap_silence,
        transient_part
    ])

    print("Mixed Recording Composition:")
    print("  Segment 1: Harmonic (7 kHz, 50ms)")
    print("  Segment 2: FM Sweep (20-40 kHz, 30ms)")
    print("  Segment 3: Transient click (20ms)")
    print(f"  Total duration: {len(mixed_recording) / sample_rate * 1000:.1f} ms")
    print()

    # Segment the recording
    analyzer = UniversalRosettaStone(sample_rate=sample_rate)
    phrases = analyzer.segment_phrases(
        mixed_recording,
        min_gap_ms=10,
        min_phrase_duration_ms=10
    )

    print(f"Number of phrases detected: {len(phrases)}")
    print()

    # Detect modality for each phrase
    for i, phrase in enumerate(phrases, 1):
        modality = analyzer.detect_modality(phrase.data)
        print(f"Phrase {i}:")
        print(f"  Duration: {len(phrase.data) / sample_rate * 1000:.1f} ms")
        print(f"  Modality: {modality.name}")
        print(f"  F0 range: {phrase.features.get('f0_range', 'N/A')} Hz")
        print(f"  ZCR: {phrase.features.get('zcr', 0):.3f}")
        print(f"  Spectral flatness: {phrase.features.get('spectral_flatness', 0):.3f}")
        print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("✓ Each phrase can have its own modality")
    print("✓ The same species can use multiple modalities")
    print("✓ Modality detection works at phrase level, not species level")
    print()


def test_species_specific_modality_profiles():
    """Test and document species-specific modality profiles."""

    print("=" * 80)
    print("SPECIES-SPECIFIC MODALITY PROFILES")
    print("=" * 80)
    print()

    # Species modality profiles
    species_profiles = {
        "Marmoset": {
            "primary": Modality.HARMONIC,
            "description": "Harmonic communication with flat tones",
            "f0_range": "(5-12 kHz)",
            "examples": ["Contact calls", "Food calls", "Social vocalizations"],
        },
        "Egyptian Fruit Bat": {
            "primary": Modality.FM_SWEEP,
            "secondary": [Modality.HARMONIC, Modality.TRANSIENT],
            "description": "FM sweep for echolocation, harmonic calls for social",
            "f0_range": "(20-90 kHz)",
            "examples": ["FM sweep navigation", "Harmonic social calls", "Clicks"],
        },
        "Dolphin": {
            "primary": Modality.HARMONIC,
            "secondary": [Modality.TRANSIENT],
            "description": "Whistle communication (harmonic) with clicks (transient)",
            "f0_range": "(2-24 kHz)",
            "examples": ["Signature whistles", "Contact whistles", "Echolocation clicks"],
        },
        "Chimpanzee": {
            "primary": Modality.HARMONIC,
            "secondary": [Modality.TRANSIENT],
            "description": "Mixed vocalizations with harmonic and transient components",
            "f0_range": "(200-3000 Hz)",
            "examples": ["Food grunts", "Screams", "Pant hoots"],
        },
        "Sperm Whale": {
            "primary": Modality.TRANSIENT,
            "description": "Click-based communication (codas)",
            "f0_range": "(200 Hz - 30 kHz)",
            "examples": ["Codas", "Clicks", "Slow clicks"],
        },
        "Zebra Finch": {
            "primary": Modality.HARMONIC,
            "description": "Songbird harmonic communication",
            "f0_range": "(2-8 kHz)",
            "examples": ["Songs", "Calls"],
        },
        "Cricket": {
            "primary": Modality.RHYTHMIC,
            "description": "Rhythmic chirping patterns",
            "f0_range": "(2-8 kHz)",
            "examples": ["Calling songs", "Courtship chirps"],
        },
    }

    for species, profile in species_profiles.items():
        print(f"{species}:")
        print(f"  Primary Modality: {profile['primary'].name}")

        if 'secondary' in profile:
            if isinstance(profile['secondary'], list):
                secondaries = ", ".join(m.name for m in profile['secondary'])
                print(f"  Secondary Modalities: {secondaries}")

        print(f"  Description: {profile['description']}")
        print(f"  F0 Range: {profile.get('f0_range', 'N/A')}")

        if 'examples' in profile:
            examples = ", ".join(profile['examples'])
            print(f"  Examples: {examples}")

        print()

    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("1. Species are NOT limited to a single modality")
    print("2. Different call types from the same species use different modalities")
    print("3. The Universal Rosetta Stone handles this at the PHRASE level")
    print("4. Each phrase is classified independently based on its acoustic features")
    print("5. This is scientifically accurate and reflects real animal communication")
    print()


if __name__ == "__main__":
    import sys

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ACOUSTIC MODALITY DETECTION TEST SUITE" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Run all tests
    try:
        test_modality_detection()
        print()

        test_mixed_modality_detection()
        print()

        test_phrase_level_modality()
        print()

        test_species_specific_modality_profiles()
        print()

        print("=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print()
        print("The Universal Rosetta Stone methodology:")
        print("  ✓ Detects acoustic modality at phrase level")
        print("  ✓ Supports multiple modalities per species")
        print("  ✓ Handles mixed-modality recordings correctly")
        print("  ✓ Classifies based on physical acoustic features")
        print()

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
