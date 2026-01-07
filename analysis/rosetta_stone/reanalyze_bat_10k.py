#!/usr/bin/env python3
"""
Re-analysis of Egyptian Fruit Bat 10k Dataset

Comprehensive analysis to confirm TRANSIENT vs FM_SWEEP modality classification.
Samples representative files from the 10,000 available.
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


def analyze_bat_file(filepath, duration_sec=2):
    """Comprehensive analysis of a single bat file."""
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

        # Get detailed modality probabilities
        probabilities = analyzer.get_modality_probabilities(audio)

        # Extract features for analysis
        features = analyzer._extract_modality_features(audio)

        # Frequency analysis
        from scipy.fft import fft, fftfreq

        fft_result = fft(audio)
        freqs = fftfreq(len(audio), 1 / sr)
        magnitude = np.abs(fft_result)

        pos_freqs = freqs[: len(freqs) // 2]
        pos_magnitude = magnitude[: len(magnitude) // 2]

        # Find dominant frequency
        dom_freq_idx = np.argmax(pos_magnitude)
        dom_freq = abs(pos_freqs[dom_freq_idx])

        # Energy bands
        bands = [
            ("Low (0-5k)", 0, 5000),
            ("Mid (5-10k)", 5000, 10000),
            ("High (10-15k)", 10000, 15000),
            ("VHF (15-20k)", 15000, 20000),
        ]

        energy_dist = {}
        total_energy = np.sum(pos_magnitude**2)
        for band_name, low, high in bands:
            mask = (pos_freqs >= low) & (pos_freqs < high)
            band_energy = np.sum(pos_magnitude[mask] ** 2)
            energy_dist[band_name] = band_energy / total_energy * 100

        # Click detection
        from scipy.signal import find_peaks, hilbert

        envelope = np.abs(hilbert(audio))
        click_threshold = np.mean(envelope) + 2 * np.std(envelope)
        peaks, _ = find_peaks(envelope, height=click_threshold, distance=int(0.005 * sr))
        click_rate = len(peaks) / (len(audio) / sr)

        # Inter-click intervals
        if len(peaks) > 1:
            intervals_ms = np.diff(peaks) / sr * 1000
            mean_ici = np.mean(intervals_ms)
            std_ici = np.std(intervals_ms)
        else:
            mean_ici = 0
            std_ici = 0

        return {
            "filename": Path(filepath).name,
            "sample_rate": sr,
            "duration_sec": len(audio) / sr,
            "overall_modality": overall_modality.name,
            "probabilities": probabilities,
            "dominant_freq_hz": dom_freq,
            "energy_distribution": energy_dist,
            "zcr": features.get("zcr", 0),
            "spectral_flatness": features.get("spectral_flatness", 0),
            "envelope_cv": features.get("envelope_cv", 0),
            "click_rate": click_rate,
            "mean_ici_ms": mean_ici,
            "std_ici_ms": std_ici,
            "num_clicks": len(peaks),
        }
    except Exception as e:
        return {"error": str(e), "filename": Path(filepath).name}


def main():
    """Re-analyze Egyptian fruit bat 10k dataset."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    bat_dir = Path.home() / "birdsong_analysis/data/egyptian_fruit_bat_10k/audio"

    if not bat_dir.exists():
        print(f"Bat data directory not found: {bat_dir}")
        return

    # Get all wav files
    all_files = sorted(list(bat_dir.glob("*.wav")))
    print(f"📁 Found {len(all_files):,} bat audio files")

    # Create representative subset
    num_to_test = 100
    if len(all_files) > num_to_test:
        # Select files evenly distributed across the range
        indices = np.linspace(0, len(all_files) - 1, num_to_test, dtype=int)
        test_files = [all_files[i] for i in indices]
    else:
        test_files = all_files

    print(f"🎲 Testing representative subset of {len(test_files)} files\n")

    print("=" * 90)
    print("EGYPTIAN FRUIT BAT 10k DATASET RE-ANALYSIS")
    print("=" * 90)

    results = []
    errors = []

    for i, filepath in enumerate(test_files):
        result = analyze_bat_file(filepath, duration_sec=2)

        if "error" in result:
            errors.append(result)
            continue

        results.append(result)

        # Print progress every 20 files
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(test_files)} files processed...")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if not results:
        print("⚠️  No successful results")
        return

    print(f"\n📊 Files successfully analyzed: {len(results)}/{len(test_files)}")

    # Overall modality distribution
    print("\n📊 OVERALL MODALITY DISTRIBUTION:")
    modality_counts = {}
    for r in results:
        m = r["overall_modality"]
        modality_counts[m] = modality_counts.get(m, 0) + 1

    for modality, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(results) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {modality:15s}: {count:3d} ({percentage:5.1f}%) {bar}")

    # Key features by modality
    print("\n📊 FEATURES BY MODALITY:")

    transients = [r for r in results if r["overall_modality"] == "TRANSIENT"]
    fm_sweeps = [r for r in results if r["overall_modality"] == "FM_SWEEP"]

    if transients:
        print(f"\n  TRANSIENT (n={len(transients)}):")
        print(
            f"    ZCR:                {np.mean([r['zcr'] for r in transients]):.4f} ± {np.std([r['zcr'] for r in transients]):.4f}"
        )
        print(
            f"    Spectral flatness:  {np.mean([r['spectral_flatness'] for r in transients]):.4f} ± {np.std([r['spectral_flatness'] for r in transients]):.4f}"
        )
        print(
            f"    Envelope CV:        {np.mean([r['envelope_cv'] for r in transients]):.4f} ± {np.std([r['envelope_cv'] for r in transients]):.4f}"
        )
        print(
            f"    Click rate:         {np.mean([r['click_rate'] for r in transients]):.1f} ± {np.std([r['click_rate'] for r in transients]):.1f} clicks/sec"
        )
        print(
            f"    Mean ICI:           {np.mean([r['mean_ici_ms'] for r in transients]):.2f} ± {np.std([r['mean_ici_ms'] for r in transients]):.2f} ms"
        )

    if fm_sweeps:
        print(f"\n  FM_SWEEP (n={len(fm_sweeps)}):")
        print(
            f"    ZCR:                {np.mean([r['zcr'] for r in fm_sweeps]):.4f} ± {np.std([r['zcr'] for r in fm_sweeps]):.4f}"
        )
        print(
            f"    Spectral flatness:  {np.mean([r['spectral_flatness'] for r in fm_sweeps]):.4f} ± {np.std([r['spectral_flatness'] for r in fm_sweeps]):.4f}"
        )
        print(
            f"    Envelope CV:        {np.mean([r['envelope_cv'] for r in fm_sweeps]):.4f} ± {np.std([r['envelope_cv'] for r in fm_sweeps]):.4f}"
        )
        print(
            f"    Click rate:         {np.mean([r['click_rate'] for r in fm_sweeps]):.1f} ± {np.std([r['click_rate'] for r in fm_sweeps]):.1f} clicks/sec"
        )

    # Energy distribution
    print("\n📊 ENERGY DISTRIBUTION (all files):")
    all_bands = ["Low (0-5k)", "Mid (5-10k)", "High (10-15k)", "VHF (15-20k)"]
    for band in all_bands:
        energies = [r["energy_distribution"].get(band, 0) for r in results]
        print(f"  {band:15s}: {np.mean(energies):5.1f}% ± {np.std(energies):.1f}%")

    # Click rate analysis
    print("\n📊 CLICK RATE ANALYSIS:")
    click_rates = [r["click_rate"] for r in results]
    print(f"  Mean:    {np.mean(click_rates):.1f} clicks/second")
    print(f"  Median:  {np.median(click_rates):.1f} clicks/second")
    print(f"  Range:   {np.min(click_rates):.1f} - {np.max(click_rates):.1f} clicks/second")
    print(f"  Std:     {np.std(click_rates):.1f} clicks/second")

    # Classification confidence
    print("\n📊 CLASSIFICATION CONFIDENCE:")

    high_confidence = 0
    low_confidence = 0

    for r in results:
        probs = r["probabilities"]
        max_prob = max(probs.values())
        if max_prob > 0.6:
            high_confidence += 1
        else:
            low_confidence += 1

    print(
        f"  High confidence (>60%): {high_confidence}/{len(results)} ({high_confidence / len(results) * 100:.1f}%)"
    )
    print(
        f"  Low confidence (≤60%):  {low_confidence}/{len(results)} ({low_confidence / len(results) * 100:.1f}%)"
    )

    # Detailed feature thresholds
    print("\n📊 MODALITY CLASSIFICATION THRESHOLDS:")

    if transients:
        print("\n  TRANSIENT files show:")
        np.mean([r["zcr"] for r in transients])
        np.mean([r["spectral_flatness"] for r in transients])
        np.mean([r["envelope_cv"] for r in transients])
        print(
            f"    ZCR < 0.1:  {sum(1 for r in transients if r['zcr'] < 0.1)}/{len(transients)} ({sum(1 for r in transients if r['zcr'] < 0.1) / len(transients) * 100:.1f}%)"
        )
        print(
            f"    Flatness < 0.3: {sum(1 for r in transients if r['spectral_flatness'] < 0.3)}/{len(transients)} ({sum(1 for r in transients if r['spectral_flatness'] < 0.3) / len(transients) * 100:.1f}%)"
        )
        print(
            f"    CV < 0.5: {sum(1 for r in transients if r['envelope_cv'] < 0.5)}/{len(transients)} ({sum(1 for r in transients if r['envelope_cv'] < 0.5) / len(transients) * 100:.1f}%)"
        )

    if fm_sweeps:
        print("\n  FM_SWEEP files show:")
        print(
            f"    ZCR > 0.1:  {sum(1 for r in fm_sweeps if r['zcr'] > 0.1)}/{len(fm_sweeps)} ({sum(1 for r in fm_sweeps if r['zcr'] > 0.1) / len(fm_sweeps) * 100:.1f}%)"
        )
        print(
            f"    Flatness < 0.6: {sum(1 for r in fm_sweeps if r['spectral_flatness'] < 0.6)}/{len(fm_sweeps)} ({sum(1 for r in fm_sweeps if r['spectral_flatness'] < 0.6) / len(fm_sweeps) * 100:.1f}%)"
        )

    # Conclusion
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    dominant_modality = max(modality_counts.items(), key=lambda x: x[1])[0]
    dominant_pct = modality_counts[dominant_modality] / len(results) * 100

    print(f"\n✅ DOMINANT MODALITY: {dominant_modality} ({dominant_pct:.1f}%)")

    if dominant_modality == "TRANSIENT":
        print("\n  This confirms the dataset is primarily TRANSIENT (click-based).")
        print("  High click rates and short ICIs indicate echolocation click trains.")
    elif dominant_modality == "FM_SWEEP":
        print("\n  The dataset is primarily FM_SWEEP (frequency-modulated signals).")
        print("  This indicates communication vocalizations rather than pure echolocation.")

    print(f"\n{'=' * 90}")
    print("✅ Re-analysis complete!")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
