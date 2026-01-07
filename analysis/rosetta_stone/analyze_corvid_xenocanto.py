#!/usr/bin/env python3
"""
Corvid Xenocanto Dataset Analysis

Comprehensive analysis of corvid vocalizations from Xenocanto:
- American Crow (Corvus brachyrhynchos) - 208 files
- Common Raven (Corvus corax) - 50 files
- Fish Crow (Corvus ossifragus) - 50 files

Total: 308 MP3 recordings from the Xenocanto database.
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


def analyze_corvid_file(filepath, duration_sec=5):
    """Comprehensive analysis of a single corvid file."""
    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Use first N seconds
        max_samples = int(duration_sec * sr)
        if len(audio) < max_samples:
            audio = audio  # Use full file if shorter
        else:
            audio = audio[:max_samples]

        analyzer = UniversalRosettaStone(sample_rate=sr)

        # Detect overall modality
        overall_modality = analyzer._detect_overall_modality(audio)

        # Get detailed modality probabilities
        probabilities = analyzer.get_modality_probabilities(audio)

        # Extract features
        features = analyzer._extract_modality_features(audio)

        # Frequency analysis
        from scipy.fft import fft, fftfreq
        fft_result = fft(audio)
        freqs = fftfreq(len(audio), 1/sr)
        magnitude = np.abs(fft_result)

        pos_freqs = freqs[:len(freqs)//2]
        pos_magnitude = magnitude[:len(magnitude)//2]

        # Find dominant frequency
        dom_freq_idx = np.argmax(pos_magnitude)
        dom_freq = abs(pos_freqs[dom_freq_idx])

        # Energy bands (corvids typically 1-8 kHz)
        bands = [
            ("Low (0-1k)", 0, 1000),
            ("Low-Mid (1-2k)", 1000, 2000),
            ("Mid (2-4k)", 2000, 4000),
            ("Mid-High (4-6k)", 4000, 6000),
            ("High (6-8k)", 6000, 8000),
            ("VHF (>8k)", 8000, sr//2),
        ]

        energy_dist = {}
        total_energy = np.sum(pos_magnitude**2)
        for band_name, low, high in bands:
            mask = (pos_freqs >= low) & (pos_freqs < high)
            band_energy = np.sum(pos_magnitude[mask]**2)
            energy_dist[band_name] = band_energy / total_energy * 100

        # Phrase segmentation with adaptive gap
        try:
            phrases = analyzer.segment_phrases(
                audio,
                min_gap_ms=50.0,
                min_phrase_duration_ms=20.0,
                use_adaptive_gap=True
            )
        except Exception:
            phrases = []

        # Get modality distribution of phrases
        phrase_modality_counts = {}
        for phrase in phrases:
            phrase_modality_counts[phrase.modality.name] = phrase_modality_counts.get(phrase.modality.name, 0) + 1

        return {
            'filename': Path(filepath).name,
            'duration_sec': len(audio) / sr,
            'sample_rate': sr,
            'overall_modality': overall_modality.name,
            'probabilities': probabilities,
            'dominant_freq_hz': dom_freq,
            'energy_distribution': energy_dist,
            'zcr': features.get('zcr', 0),
            'spectral_flatness': features.get('spectral_flatness', 0),
            'envelope_cv': features.get('envelope_cv', 0),
            'f0_mean': features.get('f0_mean', 0),
            'num_phrases': len(phrases),
            'phrase_modalities': phrase_modality_counts
        }
    except Exception as e:
        return {'error': str(e), 'filename': Path(filepath).name}


def analyze_species(species_name, species_dir, num_files=50):
    """Analyze all files for a corvid species."""
    print(f"\n{'='*90}")
    print(f"{species_name.upper()} ANALYSIS")
    print(f"{'='*90}")

    # Get all MP3 files
    all_files = sorted(list(species_dir.glob("*.mp3")))

    if len(all_files) == 0:
        print(f"⚠️  No MP3 files found in {species_dir}")
        return []

    print(f"📁 Found {len(all_files)} MP3 files")

    # Select subset if more than requested
    if len(all_files) > num_files:
        # Select evenly distributed files
        indices = np.linspace(0, len(all_files) - 1, num_files, dtype=int)
        test_files = [all_files[i] for i in indices]
    else:
        test_files = all_files

    print(f"🎲 Testing {len(test_files)} files ({len(test_files)/len(all_files)*100:.1f}% of dataset)\n")

    results = []
    errors = []

    for i, filepath in enumerate(test_files):
        result = analyze_corvid_file(filepath, duration_sec=5)

        if 'error' in result:
            errors.append(result)
            continue

        results.append(result)

        # Print progress every 10 files
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_files)} files processed...")

    return results, errors


def print_species_summary(species_name, results):
    """Print summary statistics for a species."""
    if not results:
        print(f"⚠️  No successful results for {species_name}")
        return

    print("\n📊 OVERALL MODALITY DISTRIBUTION:")
    modality_counts = {}
    for r in results:
        m = r['overall_modality']
        modality_counts[m] = modality_counts.get(m, 0) + 1

    for modality, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(results) * 100
        bar = '█' * int(percentage / 5)
        print(f"  {modality:15s}: {count:3d} ({percentage:5.1f}%) {bar}")

    # Phrase detection
    total_phrases = sum(r['num_phrases'] for r in results)
    files_with_phrases = sum(1 for r in results if r['num_phrases'] > 0)

    print("\n📊 PHRASE DETECTION:")
    print(f"  Total phrases: {total_phrases}")
    print(f"  Files with phrases: {files_with_phrases}/{len(results)} ({files_with_phrases/len(results)*100:.1f}%)")
    print(f"  Mean phrases per file: {total_phrases/len(results):.2f}")

    if files_with_phrases > 0:
        print("\n📊 DETECTED PHRASE MODALITY:")
        phrase_modality_counts = {}
        for r in results:
            for modality, count in r['phrase_modalities'].items():
                phrase_modality_counts[modality] = phrase_modality_counts.get(modality, 0) + count

        total_phrase_count = sum(phrase_modality_counts.values())
        for modality, count in sorted(phrase_modality_counts.items()):
            percentage = count / total_phrase_count * 100
            print(f"    {modality:15s}: {count:4d} ({percentage:5.1f}%)")

    # Frequency characteristics
    print("\n📊 FREQUENCY CHARACTERISTICS:")
    dom_freqs = [r['dominant_freq_hz'] for r in results if r.get('dominant_freq_hz', 0) > 0]
    if dom_freqs:
        print(f"  Dominant frequency: {np.mean(dom_freqs)/1000:.2f} ± {np.std(dom_freqs)/1000:.2f} kHz")
        print(f"  Range: {np.min(dom_freqs)/1000:.2f} - {np.max(dom_freqs)/1000:.2f} kHz")

    # Energy distribution
    print("\n📊 ENERGY DISTRIBUTION:")
    all_bands = ["Low (0-1k)", "Low-Mid (1-2k)", "Mid (2-4k)", "Mid-High (4-6k)", "High (6-8k)", "VHF (>8k)"]
    for band in all_bands:
        energies = [r['energy_distribution'].get(band, 0) for r in results]
        if energies:
            print(f"  {band:15s}: {np.mean(energies):5.1f}% ± {np.std(energies):.1f}%")


def main():
    """Analyze all corvid species from Xenocanto."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    xenocanto_dir = Path.home() / "birdsong_analysis/data/xenocanto"

    if not xenocanto_dir.exists():
        print(f"Xenocanto directory not found: {xenocanto_dir}")
        return

    print("=" * 90)
    print("CORVID XENOCANTO DATASET ANALYSIS")
    print("=" * 90)

    # Analyze each species
    all_results = {}

    # American Crow
    american_crow_dir = xenocanto_dir / "American_Crow"
    if american_crow_dir.exists():
        results, errors = analyze_species("American Crow", american_crow_dir, num_files=50)
        all_results['American_Crow'] = results
        print_species_summary("American Crow", results)

    # Common Raven
    common_raven_dir = xenocanto_dir / "Common_Raven"
    if common_raven_dir.exists():
        results, errors = analyze_species("Common Raven", common_raven_dir, num_files=30)
        all_results['Common_Raven'] = results
        print_species_summary("Common Raven", results)

    # Fish Crow
    fish_crow_dir = xenocanto_dir / "Fish_Crow"
    if fish_crow_dir.exists():
        results, errors = analyze_species("Fish Crow", fish_crow_dir, num_files=30)
        all_results['Fish_Crow'] = results
        print_species_summary("Fish Crow", results)

    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY - ALL CORVID SPECIES")
    print("=" * 90)

    total_files = sum(len(results) for results in all_results.values() if results)

    print(f"\n📊 Total files analyzed: {total_files}")

    # Cross-species modality comparison
    print("\n📊 MODALITY DISTRIBUTION BY SPECIES:")
    print(f"{'Species':<20} {'HARMONIC':>12} {'FM_SWEEP':>12} {'TRANSIENT':>12} {'RHYTHMIC':>12}")
    print("-" * 80)

    for species, results in all_results.items():
        if not results:
            continue
        modality_counts = {}
        for r in results:
            m = r['overall_modality']
            modality_counts[m] = modality_counts.get(m, 0) + 1

        n = len(results)
        h = modality_counts.get('HARMONIC', 0)
        f = modality_counts.get('FM_SWEEP', 0)
        t = modality_counts.get('TRANSIENT', 0)
        r = modality_counts.get('RHYTHMIC', 0)

        print(f"{species:<20} {h/n*100:11.1f}% {f/n*100:11.1f}% {t/n*100:11.1f}% {r/n*100:11.1f}%")

    print("\n" + "=" * 90)
    print("✅ Corvid analysis complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
