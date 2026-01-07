#!/usr/bin/env python3
"""
Test Modality Detection on Real Egyptian Fruit Bat Vocalizations

This script loads real bat vocalization data and tests the modality detection
capabilities of the Universal Rosetta Stone.

Data source: ~/birdsong_analysis/data/egyptian_fruit_bat_10k/audio/
"""

import random
import sys
from pathlib import Path

import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))  # rosetta_stone/

from universal_rosetta_stone import UniversalRosettaStone

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("⚠️  soundfile not installed. Install with: pip install soundfile")


def load_wav_file(filepath, target_sr=48000):
    """Load a WAV file and return audio as numpy array."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile library required for loading WAV files")

    try:
        audio, sr = sf.read(filepath)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            # Simple resampling (for production, use scipy.signal.resample)
            from scipy import signal as scipy_signal

            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy_signal.resample(audio, num_samples)

        return audio, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def analyze_bat_vocalization(audio_file_path, num_segments=5):
    """Analyze modality of a bat vocalization file."""
    print(f"\n{'=' * 70}")
    print(f"Analyzing: {Path(audio_file_path).name}")
    print(f"{'=' * 70}")

    # Load audio
    audio, original_sr = load_wav_file(audio_file_path)
    if audio is None:
        return None

    print(f"Original sample rate: {original_sr} Hz")
    print(f"Audio duration: {len(audio) / original_sr * 1000:.1f} ms")
    print(f"Audio samples: {len(audio)}")

    # Initialize analyzer
    analyzer = UniversalRosettaStone(sample_rate=48000)

    # Segment into phrases
    print("\n📊 Segmenting into phrases...")
    try:
        phrases = analyzer.segment_phrases(audio, min_gap_ms=10, min_phrase_duration_ms=5)
        print(f"Found {len(phrases)} phrases")
    except Exception as e:
        print(f"Error during phrase segmentation: {e}")
        return None

    if len(phrases) == 0:
        print("⚠️  No phrases detected. This file might be silent or very short.")
        return None

    # Analyze each phrase
    print(f"\n🔍 Analyzing modality for {min(num_segments, len(phrases))} phrases...")

    results = []
    for i, phrase in enumerate(phrases[:num_segments]):
        modality = analyzer.detect_modality(phrase.data)
        probabilities = analyzer.get_modality_probabilities(phrase.data)

        # Get acoustic features
        features = phrase.features

        result = {
            "phrase_num": i + 1,
            "modality": modality.name,
            "probabilities": probabilities,
            "duration_ms": features.get("duration_ms", len(phrase.data) / 48000 * 1000),
            "mean_f0_hz": features.get("f0_mean"),
            "f0_range_hz": features.get("f0_range"),
        }
        results.append(result)

        # Print result
        print(f"\n  Phrase {i + 1}:")
        print(f"    Modality: {modality.name}")
        print(f"    Probabilities: {probabilities}")
        if result["mean_f0_hz"] is not None:
            print(f"    Mean F0: {result['mean_f0_hz']:.0f} Hz")
        if result["f0_range_hz"] is not None:
            print(f"    F0 Range: {result['f0_range_hz']:.0f} Hz")
        print(f"    Duration: {result['duration_ms']:.1f} ms")

    return results


def main():
    """Main test function."""
    print("=" * 70)
    print("REAL EGYPTIAN FRUIT BAT VOCALIZATION MODALITY DETECTION TEST")
    print("=" * 70)

    # Data directory
    data_dir = Path.home() / "birdsong_analysis" / "data" / "egyptian_fruit_bat_10k" / "audio"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Get all WAV files
    wav_files = list(data_dir.glob("*.wav"))

    if len(wav_files) == 0:
        print(f"❌ No WAV files found in {data_dir}")
        return

    print(f"\n📁 Found {len(wav_files)} WAV files in {data_dir}")

    # Sample a few random files for testing
    num_files_to_test = min(5, len(wav_files))
    test_files = random.sample(wav_files, num_files_to_test)

    print(f"\n🎲 Testing {num_files_to_test} random files...")

    # Analyze each file
    all_results = []
    for wav_file in test_files:
        results = analyze_bat_vocalization(wav_file)
        if results:
            all_results.extend(results)

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    if all_results:
        # Count modality detections
        modality_counts = {}
        for result in all_results:
            modality = result["modality"]
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        print(f"\nTotal phrases analyzed: {len(all_results)}")
        print("\nModality Distribution:")
        for modality, count in sorted(modality_counts.items()):
            percentage = count / len(all_results) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {modality:15s}: {count:3d} ({percentage:5.1f}%) {bar}")

        # Expected modality for Egyptian fruit bats
        print("\n📊 Expected for Egyptian Fruit Bats:")
        print("  Primary: FM_SWEEP (for echolocation/navigation)")
        print("  Secondary: HARMONIC, TRANSIENT (for social communication)")

        print("\n✅ Test completed successfully!")
    else:
        print("\n⚠️  No results obtained. Check audio files and dependencies.")


if __name__ == "__main__":
    main()
