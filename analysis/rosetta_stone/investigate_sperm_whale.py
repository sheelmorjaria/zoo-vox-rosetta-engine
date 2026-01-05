#!/usr/bin/env python3
"""
Sperm Whale Dataset Investigation (Dominica Dataset)

Investigates sperm whale vocalizations (clicks/codas) to understand:
1. Signal characteristics and modality classification
2. Click train structure and patterns
3. Comparison with noise recordings
4. Species-specific parameters for optimal detection
"""

import numpy as np
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from universal_rosetta_stone import UniversalRosettaStone, Modality

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def load_wav_file(filepath, target_sr=None):
    """Load a WAV file - preserve native sample rate."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile library required")

    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def analyze_sperm_whale_click(filepath, is_signal=True):
    """Analyze a sperm whale file (signal or noise)."""
    file_type = "SIGNAL" if is_signal else "NOISE"

    print(f"\n{'='*70}")
    print(f"[{file_type}] {Path(filepath).name}")
    print(f"{'='*70}")

    audio, sr = load_wav_file(filepath)
    if audio is None:
        return None

    duration_ms = len(audio) / sr * 1000
    duration_sec = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration_ms:.0f} ms ({duration_sec:.2f} s)")
    print(f"Samples: {len(audio)}")
    print(f"RMS amplitude: {rms:.6f}")
    print(f"Peak amplitude: {np.max(np.abs(audio)):.6f}")

    # Frequency analysis
    from scipy.fft import fft, fftfreq
    fft_result = fft(audio)
    freqs = fftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft_result)

    pos_freqs = freqs[:len(freqs)//2]
    pos_magnitude = magnitude[:len(magnitude)//2]

    # Find dominant frequency
    dom_freq_idx = np.argmax(pos_magnitude)
    dom_freq = pos_freqs[dom_freq_idx]

    print(f"Dominant frequency: {abs(dom_freq)/1000:.1f} kHz")

    # Energy in different bands (sperm whale clicks are typically 1-15 kHz)
    bands = [
        ("Very Low (0-500 Hz)", 0, 500),
        ("Low (0.5-2 kHz)", 500, 2000),
        ("Sperm whale range (2-8 kHz)", 2000, 8000),
        ("Mid (8-15 kHz)", 8000, 15000),
        ("High (15-30 kHz)", 15000, 30000),
        ("Ultrasonic (>30 kHz)", 30000, sr//2)
    ]

    print(f"\n📊 Energy Distribution:")
    for band_name, low, high in bands:
        mask = (pos_freqs >= low) & (pos_freqs < high)
        band_energy = np.sum(pos_magnitude[mask]**2)
        total_energy = np.sum(pos_magnitude**2)
        percentage = band_energy / total_energy * 100
        if percentage > 1:
            bar = '█' * int(percentage / 3)
            print(f"  {band_name:25s}: {percentage:5.1f}% {bar}")

    # Click detection (peaks in envelope)
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(audio))

    # Set threshold based on RMS
    threshold = np.mean(envelope) + 2 * np.std(envelope)
    peaks = []
    from scipy.signal import find_peaks
    peak_indices, _ = find_peaks(envelope, height=threshold, distance=int(0.01*sr))

    print(f"\n🔍 Click Detection:")
    print(f"  Detected {len(peak_indices)} potential clicks")

    if len(peak_indices) > 0:
        # Measure inter-click intervals
        if len(peak_indices) > 1:
            intervals = np.diff(peak_indices) / sr * 1000  # Convert to ms
            print(f"  Inter-click interval: {np.mean(intervals):.1f} ± {np.std(intervals):.1f} ms")
            print(f"  Min interval: {np.min(intervals):.1f} ms")
            print(f"  Max interval: {np.max(intervals):.1f} ms")

    # Test phrase segmentation
    print(f"\n🔍 Testing phrase segmentation:")

    analyzer = UniversalRosettaStone(sample_rate=sr)

    results = {}
    for min_gap in [10, 20, 50, 100]:
        for min_dur in [5, 10, 20]:
            try:
                phrases = analyzer.segment_phrases(
                    audio,
                    min_gap_ms=min_gap,
                    min_phrase_duration_ms=min_dur
                )
                key = f"gap:{min_gap}_dur:{min_dur}"
                results[key] = len(phrases)
                if len(phrases) > 0:
                    print(f"  gap={min_gap:3d}ms, dur={min_dur:2d}ms: {len(phrases):2d} phrases ✓")
            except Exception as e:
                pass

    # Find best parameter combination
    if results:
        best_params = max(results.items(), key=lambda x: x[1])
        best_gap = int(best_params[0].split(':')[1].split('_')[0])
        best_dur = int(best_params[0].split(':')[2].split('_')[0])
        max_phrases = best_params[1]

        if max_phrases > 0:
            print(f"\n✓ Best parameters: gap={best_gap}ms, dur={best_dur}ms ({max_phrases} phrases)")

            # Analyze with best parameters
            phrases = analyzer.segment_phrases(
                audio,
                min_gap_ms=best_gap,
                min_phrase_duration_ms=best_dur
            )

            # Analyze modality for phrases
            print(f"\n📊 Modality Analysis ({len(phrases)} phrases):")
            modality_counts = {}
            details = []

            for i, phrase in enumerate(phrases[:20]):  # Max 20 phrases
                modality = analyzer.detect_modality(phrase.data)
                probabilities = analyzer.get_modality_probabilities(phrase.data)
                features = phrase.features

                modality_counts[modality.name] = modality_counts.get(modality.name, 0) + 1

                details.append({
                    'index': i,
                    'modality': modality.name,
                    'probabilities': probabilities,
                    'duration_ms': features.get('duration_ms', len(phrase.data) / sr * 1000),
                })

            # Print modality distribution
            print(f"\n  Modality Distribution:")
            for modality, count in sorted(modality_counts.items()):
                percentage = count / len(phrases) * 100
                bar = '█' * int(percentage / 10)
                expected = ""
                if modality == 'TRANSIENT':
                    expected = " (expected for clicks)"
                elif modality == 'RHYTHMIC':
                    expected = " (expected for codas)"
                print(f"    {modality:15s}: {count:2d} ({percentage:5.1f}%) {bar}{expected}")

            return {
                'filepath': filepath,
                'type': 'signal' if is_signal else 'noise',
                'total_phrases': len(phrases),
                'modality_counts': modality_counts,
                'num_clicks': len(peak_indices)
            }

    # If no segmentation, analyze whole file
    print(f"\n🔍 Analyzing entire file as single unit:")
    try:
        modality = analyzer.detect_modality(audio)
        probabilities = analyzer.get_modality_probabilities(audio)

        print(f"  Modality: {modality.name}")
        print(f"  Probabilities: {probabilities}")

        expected = ""
        if is_signal:
            if modality == 'TRANSIENT':
                expected = " ✓ (expected for clicks)"
            elif modality == 'RHYTHMIC':
                expected = " ✓ (expected for codas)"

        return {
            'filepath': filepath,
            'type': 'signal' if is_signal else 'noise',
            'total_phrases': 0,
            'full_file_modality': modality.name,
            'probabilities': probabilities,
            'num_clicks': len(peak_indices),
            'expected_check': expected
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    """Investigate sperm whale dataset."""
    print("="*70)
    print("SPERM WHALE DATASET INVESTIGATION (Dominica)")
    print("="*70)

    base_dir = Path.home() / "birdsong_analysis" / "data" / "Dominica_dataset"

    if not base_dir.exists():
        print(f"❌ Data directory not found: {base_dir}")
        return

    signal_dir = base_dir / "Signal_parts"
    noise_dir = base_dir / "Noise_parts"

    if not signal_dir.exists() or not noise_dir.exists():
        print(f"❌ Signal_parts or Noise_parts directory not found")
        return

    # Get signal and noise files
    signal_files = list(signal_dir.glob("*.wav"))
    noise_files = list(noise_dir.glob("*.wav"))

    print(f"\n📁 Found {len(signal_files)} signal files")
    print(f"📁 Found {len(noise_files)} noise files")

    # Test sample of each
    num_signal_to_test = 15
    num_noise_to_test = 5

    test_signals = random.sample(signal_files, min(num_signal_to_test, len(signal_files)))
    test_noise = random.sample(noise_files, min(num_noise_to_test, len(noise_files)))

    print(f"\n🎲 Testing {len(test_signals)} signal files and {len(test_noise)} noise files...\n")

    all_results = {'signal': [], 'noise': []}

    # Analyze signals
    print("\n" + "="*70)
    print("SIGNAL ANALYSIS (Sperm Whale Clicks/Codas)")
    print("="*70)

    for filepath in test_signals:
        result = analyze_sperm_whale_click(filepath, is_signal=True)
        if result:
            all_results['signal'].append(result)

    # Analyze noise
    print("\n" + "="*70)
    print("NOISE ANALYSIS (Background Recordings)")
    print("="*70)

    for filepath in test_noise:
        result = analyze_sperm_whale_click(filepath, is_signal=False)
        if result:
            all_results['noise'].append(result)

    # Summary
    print(f"\n{'='*70}")
    print("INVESTIGATION SUMMARY")
    print(f"{'='*70}")

    # Signal summary
    if all_results['signal']:
        print(f"\n📊 SIGNAL FILES (n={len(all_results['signal'])})")

        # Count files with phrases
        with_phrases = [r for r in all_results['signal'] if r['total_phrases'] > 0]
        without_phrases = [r for r in all_results['signal'] if r['total_phrases'] == 0]

        print(f"  With phrase segmentation: {len(with_phrases)} files")
        print(f"  Without phrase segmentation: {len(without_phrases)} files")

        # Aggregate modality for segmented files
        if with_phrases:
            total_phrases = sum(r['total_phrases'] for r in with_phrases)
            all_modality_counts = {}

            for result in with_phrases:
                for modality, count in result['modality_counts'].items():
                    all_modality_counts[modality] = all_modality_counts.get(modality, 0) + 1

            print(f"\n  Modality Distribution ({total_phrases} phrases):")
            for modality, count in sorted(all_modality_counts.items()):
                percentage = count / total_phrases * 100
                bar = '█' * int(percentage / 10)
                expected = ""
                if modality == 'TRANSIENT':
                    expected = " ✓ (clicks)"
                elif modality == 'RHYTHMIC':
                    expected = " ✓ (codas)"
                print(f"    {modality:15s}: {count:3d} ({percentage:5.1f}%) {bar}{expected}")

        # Full-file modality for non-segmented
        if without_phrases:
            print(f"\n  Full-file Modality (no segmentation):")
            modality_counts = {}
            for result in without_phrases:
                m = result['full_file_modality']
                modality_counts[m] = modality_counts.get(m, 0) + 1

            for modality, count in sorted(modality_counts.items()):
                percentage = count / len(without_phrases) * 100
                bar = '█' * int(percentage / 10)
                print(f"    {modality:15s}: {count:2d} ({percentage:5.1f}%) {bar}")

    # Noise summary
    if all_results['noise']:
        print(f"\n📊 NOISE FILES (n={len(all_results['noise'])})")

        modality_counts = {}
        for result in all_results['noise']:
            if result['total_phrases'] == 0:
                m = result['full_file_modality']
            else:
                m = list(result['modality_counts'].keys())[0]  # Primary modality
            modality_counts[m] = modality_counts.get(m, 0) + 1

        print(f"  Modality Distribution:")
        for modality, count in sorted(modality_counts.items()):
            percentage = count / len(all_results['noise']) * 100
            bar = '█' * int(percentage / 10)
            print(f"    {modality:15s}: {count:2d} ({percentage:5.1f}%) {bar}")

    print(f"\n💡 Key Findings:")
    print(f"  - Sample rate: 156.25 kHz (ultrasonic)")
    print(f"  - Sperm whale clicks: 1-15 kHz range")
    print(f"  - Expected: TRANSIENT (clicks) or RHYTHMIC (codas)")

    print(f"\n{'='*70}")
    print("✅ Investigation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
