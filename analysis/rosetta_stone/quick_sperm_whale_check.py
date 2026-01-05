#!/usr/bin/env python3
"""
Quick Sperm Whale Check
Analyzes first 10 seconds of each file
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from universal_rosetta_stone import UniversalRosettaStone, Modality

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def quick_check(filepath, max_duration_sec=10):
    """Quick check of first N seconds."""
    try:
        # Get sample rate first
        info = sf.info(filepath)
        sr = info.samplerate
        audio, sr = sf.read(filepath, frames=int(max_duration_sec * sr))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None

    duration_ms = len(audio) / sr * 1000
    rms = np.sqrt(np.mean(audio**2))

    # Energy in bands
    from scipy.fft import fft, fftfreq
    fft_result = fft(audio)
    freqs = fftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft_result)

    pos_freqs = freqs[:len(freqs)//2]
    pos_magnitude = magnitude[:len(magnitude)//2]

    bands = [
        ("0-2 kHz", 0, 2000),
        ("2-8 kHz (SW clicks)", 2000, 8000),
        ("8-15 kHz", 8000, 15000),
        (">15 kHz", 15000, sr//2)
    ]

    energy_dist = {}
    total_energy = np.sum(pos_magnitude**2)
    for band_name, low, high in bands:
        mask = (pos_freqs >= low) & (pos_freqs < high)
        band_energy = np.sum(pos_magnitude[mask]**2)
        energy_dist[band_name] = band_energy / total_energy * 100

    # Detect modality
    analyzer = UniversalRosettaStone(sample_rate=sr)

    try:
        phrases = analyzer.segment_phrases(audio, min_gap_ms=50, min_phrase_duration_ms=10)
    except:
        phrases = []

    if len(phrases) > 0:
        modality_counts = {}
        for phrase in phrases[:10]:
            m = analyzer.detect_modality(phrase.data)
            modality_counts[m.name] = modality_counts.get(m.name, 0) + 1
        return {
            'name': Path(filepath).name,
            'duration_ms': duration_ms,
            'rms': rms,
            'energy': energy_dist,
            'phrases': len(phrases),
            'modality_counts': modality_counts
        }
    else:
        # Analyze whole file
        m = analyzer.detect_modality(audio)
        probs = analyzer.get_modality_probabilities(audio)
        return {
            'name': Path(filepath).name,
            'duration_ms': duration_ms,
            'rms': rms,
            'energy': energy_dist,
            'phrases': 0,
            'modality': m.name,
            'probabilities': probs
        }


def main():
    print("="*70)
    print("QUICK SPERM WHALE CHECK (first 10 seconds only)")
    print("="*70)

    base_dir = Path.home() / "birdsong_analysis" / "data" / "Dominica_dataset"
    signal_dir = base_dir / "Signal_parts"

    signal_files = list(signal_dir.glob("*.wav"))

    print(f"\n📁 Found {len(signal_files)} signal files")
    print(f"🎲 Testing first 10 files (10 seconds each)...\n")

    results = []
    for filepath in signal_files[:10]:
        r = quick_check(filepath)
        if r:
            results.append(r)
            print(f"✓ {r['name'][:30]:30s} {r['duration_ms']:7.0f}ms  RMS:{r['rms']:.4f}  Phrases:{r['phrases']:2d}  ", end="")

            if r['phrases'] > 0:
                print(f"Modalities: {r['modality_counts']}")
            else:
                print(f"Modality: {r['modality']}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    with_phrases = [r for r in results if r['phrases'] > 0]
    without_phrases = [r for r in results if r['phrases'] == 0]

    print(f"\nFiles with phrase segmentation: {len(with_phrases)}/{len(results)}")
    print(f"Files without segmentation: {len(without_phrases)}/{len(results)}")

    # Energy averages
    print(f"\n📊 Average Energy Distribution:")
    avg_energy = {}
    for band in ["0-2 kHz", "2-8 kHz (SW clicks)", "8-15 kHz", ">15 kHz"]:
        avg_energy[band] = np.mean([r['energy'].get(band, 0) for r in results])

    for band, avg in avg_energy.items():
        print(f"  {band:20s}: {avg:5.1f}%")

    # Modality summary
    if with_phrases:
        print(f"\n📊 Modality Distribution (segmented files):")
        all_counts = {}
        for r in with_phrases:
            for m, c in r['modality_counts'].items():
                all_counts[m] = all_counts.get(m, 0) + c

        for m, c in sorted(all_counts.items()):
            print(f"  {m:15s}: {c}")

    if without_phrases:
        print(f"\n📊 Full-file Modality (no segmentation):")
        counts = {}
        for r in without_phrases:
            m = r['modality']
            counts[m] = counts.get(m, 0) + 1

        for m, c in sorted(counts.items()):
            print(f"  {m:15s}: {c}")

    print(f"\n✅ Check complete!")


if __name__ == "__main__":
    if HAS_SOUNDFILE:
        main()
