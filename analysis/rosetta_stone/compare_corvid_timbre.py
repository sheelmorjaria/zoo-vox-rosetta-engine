#!/usr/bin/env python3
"""
Timbre Feature Comparison: Common Raven vs. Fish Crow

This script tests whether the new timbre features (spectral centroid, slope,
bandwidth, and rolloff) can distinguish between Common Raven and Fish Crow,
which previously showed identical statistics.

Category 1, Item 1: Spectral Centroid & Slope (The "Timbre" Fix)
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def analyze_corvid_timbre(filepath, duration_sec=5):
    """Comprehensive timbre analysis of a single corvid file."""
    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Use first N seconds
        max_samples = int(duration_sec * sr)
        if len(audio) < max_samples:
            audio = audio
        else:
            audio = audio[:max_samples]

        analyzer = UniversalRosettaStone(sample_rate=sr)

        # Extract features including timbre
        features = analyzer._extract_modality_features(audio)

        # Get overall modality
        overall_modality = analyzer._detect_overall_modality(audio)

        return {
            'filename': Path(filepath).name,
            'duration_sec': len(audio) / sr,
            'sample_rate': sr,
            'overall_modality': overall_modality.name,
            # Timbre features (Category 1, Item 1)
            'spectral_centroid_hz': features.get('spectral_centroid_hz', 0),
            'spectral_slope': features.get('spectral_slope', 0),
            'spectral_bandwidth_hz': features.get('spectral_bandwidth_hz', 0),
            'spectral_rolloff_hz': features.get('spectral_rolloff_hz', 0),
            # Other features for context
            'zcr': features.get('zcr', 0),
            'spectral_flatness': features.get('spectral_flatness', 0),
            'envelope_cv': features.get('envelope_cv', 0),
        }
    except Exception as e:
        return {'error': str(e), 'filename': Path(filepath).name}


def compare_species_timbre(species1_name, species1_dir, species2_name, species2_dir, num_files=30):
    """Compare timbre features between two corvid species."""
    print(f"\n{'='*90}")
    print(f"TIMBRE COMPARISON: {species1_name} vs {species2_name}")
    print(f"{'='*90}\n")

    # Load species 1
    all_files1 = sorted(list(species1_dir.glob("*.mp3")))
    if len(all_files1) > num_files:
        indices = np.linspace(0, len(all_files1) - 1, num_files, dtype=int)
        test_files1 = [all_files1[i] for i in indices]
    else:
        test_files1 = all_files1

    # Load species 2
    all_files2 = sorted(list(species2_dir.glob("*.mp3")))
    if len(all_files2) > num_files:
        indices = np.linspace(0, len(all_files2) - 1, num_files, dtype=int)
        test_files2 = [all_files2[i] for i in indices]
    else:
        test_files2 = all_files2

    print(f"📁 {species1_name}: {len(test_files1)} files")
    print(f"📁 {species2_name}: {len(test_files2)} files\n")

    # Analyze species 1
    print(f"Analyzing {species1_name}...")
    results1 = []
    for filepath in test_files1:
        result = analyze_corvid_timbre(filepath, duration_sec=5)
        if 'error' not in result:
            results1.append(result)

    # Analyze species 2
    print(f"Analyzing {species2_name}...")
    results2 = []
    for filepath in test_files2:
        result = analyze_corvid_timbre(filepath, duration_sec=5)
        if 'error' not in result:
            results2.append(result)

    if not results1 or not results2:
        print("⚠️  Insufficient data for comparison")
        return

    # Comparison statistics
    print(f"\n{'='*90}")
    print(f"TIMBRE FEATURE STATISTICS")
    print(f"{'='*90}\n")

    timbre_features = [
        'spectral_centroid_hz',
        'spectral_slope',
        'spectral_bandwidth_hz',
        'spectral_rolloff_hz',
    ]

    print(f"{'Feature':<25} {species1_name:<20} {species2_name:<20} {'p-value':<12} {'Significant?':<12}")
    print("-" * 90)

    significant_features = []

    for feature in timbre_features:
        values1 = [r[feature] for r in results1]
        values2 = [r[feature] for r in results2]

        mean1 = np.mean(values1)
        std1 = np.std(values1)
        mean2 = np.mean(values2)
        std2 = np.std(values2)

        # Statistical test (Mann-Whitney U test for non-normal distributions)
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')

        # Significance threshold
        is_significant = p_value < 0.05
        if is_significant:
            significant_features.append((feature, p_value))

        sig_marker = "✅ YES" if is_significant else "❌ NO"

        print(f"{feature:<25} {mean1:>8.2f} ± {std1:<6.2f} {mean2:>8.2f} ± {std2:<6.2f} {p_value:<12.4f} {sig_marker:<12}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if significant_features:
        print(f"\n✅ {len(significant_features)} timbre features significantly different (p < 0.05):")
        for feature, p_value in significant_features:
            print(f"  - {feature}: p = {p_value:.4f}")
        print(f"\n✅ TIMBRE FEATURES CAN DISTINGUISH {species1_name.upper()} FROM {species2_name.upper()}")
    else:
        print(f"\n⚠️  No timbre features significantly different")
        print(f"⚠️  TIMBRE FEATURES CANNOT DISTINGUISH THESE SPECIES")

    # Effect size (Cohen's d for most significant feature)
    if significant_features:
        best_feature, _ = significant_features[0]
        values1 = [r[best_feature] for r in results1]
        values2 = [r[best_feature] for r in results2]

        pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
        cohens_d = abs(np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0

        print(f"\n📊 Effect size (Cohen's d) for {best_feature}: {cohens_d:.3f}")
        if cohens_d < 0.2:
            print("  Effect size: Small")
        elif cohens_d < 0.5:
            print("  Effect size: Medium")
        elif cohens_d < 0.8:
            print("  Effect size: Large")
        else:
            print("  Effect size: Very large")

    # Modality distribution comparison
    print(f"\n📊 MODALITY DISTRIBUTION:")

    modality_counts1 = {}
    for r in results1:
        m = r['overall_modality']
        modality_counts1[m] = modality_counts1.get(m, 0) + 1

    modality_counts2 = {}
    for r in results2:
        m = r['overall_modality']
        modality_counts2[m] = modality_counts2.get(m, 0) + 1

    print(f"\n  {species1_name}:")
    for modality, count in sorted(modality_counts1.items()):
        percentage = count / len(results1) * 100
        print(f"    {modality}: {count} ({percentage:.1f}%)")

    print(f"\n  {species2_name}:")
    for modality, count in sorted(modality_counts2.items()):
        percentage = count / len(results2) * 100
        print(f"    {modality}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 90)


def main():
    """Main analysis comparing Common Raven and Fish Crow timbre."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    xenocanto_dir = Path.home() / "birdsong_analysis/data/xenocanto"

    if not xenocanto_dir.exists():
        print(f"Xenocanto directory not found: {xenocanto_dir}")
        return

    # Compare Common Raven vs. Fish Crow
    common_raven_dir = xenocanto_dir / "Common_Raven"
    fish_crow_dir = xenocanto_dir / "Fish_Crow"

    if common_raven_dir.exists() and fish_crow_dir.exists():
        compare_species_timbre(
            "Common Raven",
            common_raven_dir,
            "Fish Crow",
            fish_crow_dir,
            num_files=30
        )
    else:
        print("⚠️  Species directories not found")

    print("\n" + "=" * 90)
    print("✅ Timbre comparison complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
