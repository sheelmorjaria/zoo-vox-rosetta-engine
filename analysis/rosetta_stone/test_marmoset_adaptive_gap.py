#!/usr/bin/env python3
"""
Marmoset Adaptive Gap Test

Tests adaptive gap threshold on marmoset vocalizations.
Uses a representative subset of the 871,045 available FLAC files.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def test_marmoset_file(filepath, duration_sec=2):
    """Test adaptive gap on a single marmoset file."""
    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Use first N seconds
        max_samples = int(duration_sec * sr)
        audio = audio[:max_samples]

        analyzer = UniversalRosettaStone(sample_rate=sr)

        # Detect overall modality
        overall_modality = analyzer._detect_overall_modality(audio)

        # Calculate adaptive threshold
        adaptive_threshold = analyzer._calculate_adaptive_gap_threshold(audio)

        # For HARMONIC signals, adaptive gap should have minimal effect
        # Test with adaptive gap
        phrases_adaptive = analyzer.segment_phrases(
            audio,
            min_gap_ms=30.0,  # Use 30ms for marmosets (optimized parameter)
            use_adaptive_gap=True
        )

        # Test without adaptive gap
        phrases_fixed = analyzer.segment_phrases(
            audio,
            min_gap_ms=30.0,
            use_adaptive_gap=False
        )

        # Get modality distribution
        modality_counts = {}
        for phrase in phrases_adaptive:
            modality_counts[phrase.modality.name] = modality_counts.get(phrase.modality.name, 0) + 1

        return {
            'filename': Path(filepath).name,
            'duration_sec': len(audio) / sr,
            'sample_rate': sr,
            'overall_modality': overall_modality.name,
            'adaptive_threshold_ms': adaptive_threshold,
            'phrases_adaptive': len(phrases_adaptive),
            'phrases_fixed': len(phrases_fixed),
            'improvement': len(phrases_adaptive) - len(phrases_fixed),
            'modality_distribution': modality_counts
        }
    except Exception as e:
        return {'error': str(e), 'filename': Path(filepath).name}


def main():
    """Test adaptive gap on representative marmoset subset."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    # Find marmoset data directory
    vocalizations_dir = Path.home() / "birdsong_analysis/data/Vocalizations"

    if not vocalizations_dir.exists():
        print(f"Marmoset data directory not found: {vocalizations_dir}")
        return

    # Get all FLAC files from subdirectories
    all_files = sorted(list(vocalizations_dir.glob("**/*.flac")))
    print(f"📁 Found {len(all_files):,} marmoset vocalization files")

    # Create representative subset (sample evenly across directories)
    num_to_test = 50

    # Get unique subdirectories
    subdirs = sorted(list(set([f.parent for f in all_files])))
    print(f"📁 Found {len(subdirs)} subdirectories")

    # Sample files evenly across subdirectories
    test_files = []
    files_per_subdir = max(1, num_to_test // len(subdirs))

    for subdir in subdirs[:num_to_test]:  # Limit to first N subdirs
        files_in_subdir = sorted(list(subdir.glob("*.flac")))
        if files_in_subdir:
            # Take up to files_per_subdir from each
            test_files.extend(files_in_subdir[:files_per_subdir])
            if len(test_files) >= num_to_test:
                break

    # Limit to num_to_test
    test_files = test_files[:num_to_test]

    print(f"🎲 Testing representative subset of {len(test_files)} files\n")

    print("=" * 80)
    print("MARMOSET ADAPTIVE GAP TEST")
    print("=" * 80)

    results = []
    errors = []

    for i, filepath in enumerate(test_files):
        result = test_marmoset_file(filepath, duration_sec=2)

        if 'error' in result:
            errors.append(result)
            continue

        results.append(result)

        # Print progress every 10 files
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(test_files)} files processed...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results:
        print("⚠️  No successful results")
        if errors:
            print(f"\nErrors encountered: {len(errors)}")
            for e in errors[:5]:
                print(f"  {e['filename']}: {e['error']}")
        return

    print(f"\n📊 Files successfully analyzed: {len(results)}/{len(test_files)}")

    # Overall statistics
    print("\n📊 MODALITY DISTRIBUTION:")
    modality_counts = {}
    for r in results:
        m = r['overall_modality']
        modality_counts[m] = modality_counts.get(m, 0) + 1

    for modality, count in sorted(modality_counts.items()):
        percentage = count / len(results) * 100
        print(f"  {modality:15s}: {count:3d} ({percentage:5.1f}%)")

    print("\n📊 ADAPTIVE THRESHOLD STATISTICS:")
    thresholds = [r['adaptive_threshold_ms'] for r in results]
    print(f"  Mean: {np.mean(thresholds):.2f} ms")
    print(f"  Median: {np.median(thresholds):.2f} ms")
    print(f"  Range: {np.min(thresholds):.2f} - {np.max(thresholds):.2f} ms")
    print(f"  Std: {np.std(thresholds):.2f} ms")

    print("\n📊 PHRASE DETECTION:")
    total_phrases_adaptive = sum(r['phrases_adaptive'] for r in results)
    total_phrases_fixed = sum(r['phrases_fixed'] for r in results)
    files_with_phrases_adaptive = sum(1 for r in results if r['phrases_adaptive'] > 0)
    files_with_phrases_fixed = sum(1 for r in results if r['phrases_fixed'] > 0)

    print(f"  Total phrases (adaptive): {total_phrases_adaptive}")
    print(f"  Total phrases (fixed 30ms): {total_phrases_fixed}")
    print(f"  Files with phrases (adaptive): {files_with_phrases_adaptive}/{len(results)} ({files_with_phrases_adaptive/len(results)*100:.1f}%)")
    print(f"  Files with phrases (fixed): {files_with_phrases_fixed}/{len(results)} ({files_with_phrases_fixed/len(results)*100:.1f}%)")

    # Improvement analysis
    improvements = [r['improvement'] for r in results]
    files_with_improvement = sum(1 for i in improvements if i > 0)
    files_with_decrease = sum(1 for i in improvements if i < 0)
    files_same = sum(1 for i in improvements if i == 0)
    total_improvement = sum(improvements)

    print("\n📊 ADAPTIVE GAP EFFECT:")
    print(f"  Files with improvement: {files_with_improvement}/{len(results)} ({files_with_improvement/len(results)*100:.1f}%)")
    print(f"  Files with decrease: {files_with_decrease}/{len(results)} ({files_with_decrease/len(results)*100:.1f}%)")
    print(f"  Files unchanged: {files_same}/{len(results)} ({files_same/len(results)*100:.1f}%)")
    print(f"  Net change: {total_improvement:+d} phrases")

    # For HARMONIC signals, adaptive should have minimal effect
    harmonic_results = [r for r in results if r['overall_modality'] == 'HARMONIC']
    if harmonic_results:
        print(f"\n📊 HARMONIC FILES (n={len(harmonic_results)}):")
        harmonic_adaptive = sum(r['phrases_adaptive'] for r in harmonic_results)
        harmonic_fixed = sum(r['phrases_fixed'] for r in harmonic_results)
        print(f"  Phrases (adaptive): {harmonic_adaptive}")
        print(f"  Phrases (fixed): {harmonic_fixed}")
        if harmonic_fixed > 0:
            diff_pct = ((harmonic_adaptive - harmonic_fixed) / harmonic_fixed) * 100
            print(f"  Difference: {diff_pct:+.1f}%")

    # Modality distribution of detected phrases
    print("\n📊 DETECTED PHRASE MODALITY:")
    phrase_modality_counts = {}
    for r in results:
        for modality, count in r['modality_distribution'].items():
            phrase_modality_counts[modality] = phrase_modality_counts.get(modality, 0) + count

    total_detected_phrases = sum(phrase_modality_counts.values())
    if total_detected_phrases > 0:
        for modality, count in sorted(phrase_modality_counts.items()):
            percentage = count / total_detected_phrases * 100
            print(f"  {modality:15s}: {count:4d} ({percentage:5.1f}%)")

    print("\n" + "=" * 80)
    print("✅ Marmoset test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
