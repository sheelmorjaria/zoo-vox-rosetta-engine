#!/usr/bin/env python3
"""
Cross-Species Adaptive Gap Threshold Test

Tests the adaptive gap enhancement across multiple species to verify
it works correctly for different vocalization types.
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


def test_file(filepath, duration_sec=10, expected_modality=None):
    """Test adaptive gap on a single file."""
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Use first N seconds
    audio = audio[:int(duration_sec * sr)]

    analyzer = UniversalRosettaStone(sample_rate=sr)

    # Detect overall modality
    overall_modality = analyzer._detect_overall_modality(audio)

    # Calculate adaptive threshold
    adaptive_threshold = analyzer._calculate_adaptive_gap_threshold(audio)

    # Test with adaptive gap
    phrases_adaptive = analyzer.segment_phrases(
        audio,
        min_gap_ms=50.0,
        use_adaptive_gap=True
    )

    # Test without adaptive gap
    phrases_fixed = analyzer.segment_phrases(
        audio,
        min_gap_ms=50.0,
        use_adaptive_gap=False
    )

    # Get modality distribution of detected phrases
    modality_counts = {}
    for phrase in phrases_adaptive:
        modality_counts[phrase.modality.name] = modality_counts.get(phrase.modality.name, 0) + 1

    return {
        'filename': Path(filepath).name,
        'species': filepath.parent.parent.name,
        'overall_modality': overall_modality.name,
        'expected_modality': expected_modality,
        'adaptive_threshold_ms': adaptive_threshold,
        'phrases_adaptive': len(phrases_adaptive),
        'phrases_fixed': len(phrases_fixed),
        'improvement': len(phrases_adaptive) - len(phrases_fixed),
        'modality_distribution': modality_counts
    }


def test_marmoset_files(base_dir, num_files=5):
    """Test marmoset files (HARMONIC - should NOT use adaptive gap)."""
    print("\n" + "="*70)
    print("MARMOSET TESTS (Expected: HARMONIC)")
    print("="*70)

    marmoset_dir = base_dir / "marmoset_data"

    # Try different possible locations
    possible_paths = [
        Path.home() / "birdsong_analysis/data/marmoset_data",
        Path.home() / "birdsong_analysis/data/Marmoset",
        Path.cwd().parent / "data" / "marmoset_data"
    ]

    marmoset_dir = None
    for path in possible_paths:
        if path.exists():
            marmoset_dir = path
            break

    if not marmoset_dir:
        print("⚠️  Marmoset data directory not found")
        return []

    # Find WAV files
    wav_files = list(marmoset_dir.glob("*.wav"))[:num_files]

    if not wav_files:
        # Try subdirectories
        for subdir in marmoset_dir.iterdir():
            if subdir.is_dir():
                wav_files = list(subdir.glob("*.wav"))[:num_files]
                if wav_files:
                    break

    if not wav_files:
        print("⚠️  No marmoset WAV files found")
        return []

    results = []
    for filepath in wav_files:
        try:
            result = test_file(filepath, duration_sec=5, expected_modality="HARMONIC")
            results.append(result)

            print(f"\n{result['filename'][:40]:40s}")
            print(f"  Modality: {result['overall_modality']} {'✓' if result['overall_modality'] == 'HARMONIC' else '✗'}")
            print(f"  Adaptive threshold: {result['adaptive_threshold_ms']:6.2f} ms")
            print(f"  Phrases: {result['phrases_adaptive']} (adaptive) vs {result['phrases_fixed']} (fixed)")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def test_bat_files(base_dir, num_files=5):
    """Test Egyptian fruit bat files (TRANSIENT - should use adaptive gap)."""
    print("\n" + "="*70)
    print("EGYPTIAN FRUIT BAT TESTS (Expected: TRANSIENT)")
    print("="*70)

    bat_dir = base_dir / "egyptian_fruit_bat_10k"

    # Try different possible locations
    possible_paths = [
        Path.home() / "birdsong_analysis/data/egyptian_fruit_bat_10k",
        Path.cwd().parent / "data" / "egyptian_fruit_bat_10k"
    ]

    bat_dir = None
    for path in possible_paths:
        if path.exists():
            bat_dir = path
            break

    if not bat_dir:
        print("⚠️  Bat data directory not found")
        return []

    wav_files = list(bat_dir.glob("*.wav"))[:num_files]

    if not wav_files:
        print("⚠️  No bat WAV files found")
        return []

    results = []
    for filepath in wav_files:
        try:
            result = test_file(filepath, duration_sec=5, expected_modality="TRANSIENT")
            results.append(result)

            print(f"\n{result['filename'][:40]:40s}")
            print(f"  Modality: {result['overall_modality']} {'✓' if result['overall_modality'] == 'TRANSIENT' else '✗'}")
            print(f"  Adaptive threshold: {result['adaptive_threshold_ms']:6.2f} ms")
            print(f"  Phrases: {result['phrases_adaptive']} (adaptive) vs {result['phrases_fixed']} (fixed)")
            if result['improvement'] > 0:
                print(f"  Improvement: +{result['improvement']} phrases")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def test_dolphin_files(base_dir, num_files=5):
    """Test dolphin files (HARMONIC/FM - mixed results)."""
    print("\n" + "="*70)
    print("BOTTLENOSE DOLPHIN TESTS (Expected: HARMONIC/FM_SWEEP)")
    print("="*70)

    dolphin_dir = Path.home() / "birdsong_analysis/data/Whistle_Signals"

    if not dolphin_dir.exists():
        print("⚠️  Dolphin data directory not found")
        return []

    wav_files = list(dolphin_dir.glob("**/*.wav"))[:num_files]

    if not wav_files:
        print("⚠️  No dolphin WAV files found")
        return []

    results = []
    for filepath in wav_files:
        try:
            result = test_file(filepath, duration_sec=5, expected_modality="HARMONIC")
            results.append(result)

            print(f"\n{result['filename'][:40]:40s}")
            print(f"  Modality: {result['overall_modality']}")
            print(f"  Adaptive threshold: {result['adaptive_threshold_ms']:6.2f} ms")
            print(f"  Phrases: {result['phrases_adaptive']} (adaptive) vs {result['phrases_fixed']} (fixed)")
            if result['modality_distribution']:
                print(f"  Modality distribution: {result['modality_distribution']}")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def test_sperm_whale_files(base_dir, num_files=5):
    """Test sperm whale files (TRANSIENT - should use adaptive gap)."""
    print("\n" + "="*70)
    print("SPERM WHALE TESTS (Expected: TRANSIENT)")
    print("="*70)

    whale_dir = base_dir / "Dominica_dataset/Signal_parts"

    if not whale_dir.exists():
        print("⚠️  Sperm whale data directory not found")
        return []

    wav_files = sorted(list(whale_dir.glob("*.wav")))[:num_files]

    if not wav_files:
        print("⚠️  No sperm whale WAV files found")
        return []

    results = []
    for filepath in wav_files:
        try:
            result = test_file(filepath, duration_sec=10, expected_modality="TRANSIENT")
            results.append(result)

            print(f"\n{result['filename'][:40]:40s}")
            print(f"  Modality: {result['overall_modality']} {'✓' if result['overall_modality'] == 'TRANSIENT' else '✗'}")
            print(f"  Adaptive threshold: {result['adaptive_threshold_ms']:6.2f} ms")
            print(f"  Phrases: {result['phrases_adaptive']} (adaptive) vs {result['phrases_fixed']} (fixed)")
            if result['improvement'] > 0:
                print(f"  Improvement: +{result['improvement']} phrases")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def main():
    """Run cross-species tests."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    base_dir = Path.home() / "birdsong_analysis/data"

    print("="*70)
    print("CROSS-SPECIES ADAPTIVE GAP TEST")
    print("="*70)

    all_results = {}

    # Test each species
    all_results['marmoset'] = test_marmoset_files(base_dir, num_files=3)
    all_results['bat'] = test_bat_files(base_dir, num_files=3)
    all_results['dolphin'] = test_dolphin_files(base_dir, num_files=3)
    all_results['sperm_whale'] = test_sperm_whale_files(base_dir, num_files=3)

    # Summary
    print("\n" + "="*70)
    print("CROSS-SPECIES SUMMARY")
    print("="*70)

    for species, results in all_results.items():
        if not results:
            continue

        print(f"\n{species.upper()}:")
        print(f"  Files tested: {len(results)}")

        modalities = [r['overall_modality'] for r in results]
        from collections import Counter
        modality_counts = Counter(modalities)
        print(f"  Modality distribution: {dict(modality_counts)}")

        total_phrases_adaptive = sum(r['phrases_adaptive'] for r in results)
        total_phrases_fixed = sum(r['phrases_fixed'] for r in results)
        print(f"  Total phrases: {total_phrases_adaptive} (adaptive) vs {total_phrases_fixed} (fixed)")

        improvements = [r['improvement'] for r in results]
        files_with_improvement = sum(1 for i in improvements if i > 0)
        print(f"  Files with improvement: {files_with_improvement}/{len(results)}")

        if species in ['bat', 'sperm_whale']:
            # TRANSIENT species should show improvement
            if files_with_improvement == len(results):
                print(f"  ✅ Excellent: All files show improvement")
            elif files_with_improvement > 0:
                print(f"  ~ Partial: Some files show improvement")
            else:
                print(f"  ✗ Poor: No files show improvement")
        elif species == 'marmoset':
            # HARMONIC species should work similarly with both methods
            diff = abs(total_phrases_adaptive - total_phrases_fixed)
            if diff == 0:
                print(f"  ✅ Excellent: Same results (adaptive correctly not applied)")
            elif diff < total_phrases_adaptive * 0.1:
                print(f"  ✅ Good: Minimal difference")
            else:
                print(f"  ~ Different results (may need tuning)")

    print("\n" + "="*70)
    print("✅ Cross-species test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
