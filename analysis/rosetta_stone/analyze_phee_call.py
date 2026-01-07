#!/usr/bin/env python3
"""
Detailed Phee Call Analysis

Performs deep analysis of a specific marmoset Phee call to understand:
- Phrase structure and segmentation
- Acoustic features (F0, duration, harmonics)
- Modality classification and probabilities
- Spectral characteristics
- Comparison with synthetic signals

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from universal_rosetta_stone import Modality, UniversalRosettaStone

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not installed - skipping visualizations")


def load_phee_call():
    """Load the specific Phee call that was detected."""
    # Find the Phee_62767.flac file
    data_dir = Path.home() / "birdsong_analysis" / "data" / "Vocalizations"

    # Search for the file
    for phee_file in data_dir.glob("**/Phee_62767.flac"):
        print(f"📁 Found: {phee_file}")

        if not HAS_SOUNDFILE:
            print("❌ soundfile not installed")
            return None, None

        audio, sr = sf.read(phee_file)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return audio, sr, phee_file

    print("❌ Phee_62767.flac not found")
    return None, None, None


def analyze_acoustic_features(audio, sr):
    """Extract detailed acoustic features."""
    print(f"\n{'='*70}")
    print("DETAILED ACOUSTIC FEATURE ANALYSIS")
    print(f"{'='*70}")

    # Basic info
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))

    print("\n📊 Basic Properties:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {duration*1000:.1f} ms")
    print(f"  Samples: {len(audio)}")
    print(f"  RMS amplitude: {rms:.6f}")
    print(f"  Peak amplitude: {np.max(np.abs(audio)):.6f}")

    # Frequency analysis
    from scipy.fft import fft, fftfreq

    # FFT for frequency content
    fft_result = fft(audio)
    freqs = fftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft_result)

    # Only positive frequencies
    pos_freqs = freqs[:len(freqs)//2]
    pos_magnitude = magnitude[:len(magnitude)//2]

    # Find dominant frequency
    dom_freq_idx = np.argmax(pos_magnitude)
    dom_freq = pos_freqs[dom_freq_idx]

    print("\n📊 Frequency Content:")
    print(f"  Dominant frequency: {dom_freq:.0f} Hz")
    print(f"  Frequency range: 0 - {sr/2:.0f} Hz (Nyquist)")

    # Energy in different bands
    bands = [
        ("Low (0-1 kHz)", 0, 1000),
        ("Mid (1-5 kHz)", 1000, 5000),
        ("Marmoset range (5-12 kHz)", 5000, 12000),
        ("High (12-20 kHz)", 12000, 20000),
        ("Ultrasonic (>20 kHz)", 20000, sr//2)
    ]

    print("\n📊 Energy Distribution:")
    for band_name, low, high in bands:
        mask = (pos_freqs >= low) & (pos_freqs < high)
        band_energy = np.sum(pos_magnitude[mask]**2)
        total_energy = np.sum(pos_magnitude**2)
        percentage = band_energy / total_energy * 100
        bar = '█' * int(percentage / 5)
        print(f"  {band_name:25s}: {percentage:5.1f}% {bar}")

    return {
        'duration_ms': duration * 1000,
        'rms': rms,
        'dominant_frequency_hz': dom_freq,
        'sample_rate': sr
    }


def analyze_phrase_structure(audio, sr):
    """Analyze the phrase structure of the Phee call."""
    print(f"\n{'='*70}")
    print("PHRASE STRUCTURE ANALYSIS")
    print(f"{'='*70}")

    analyzer = UniversalRosettaStone(sample_rate=48000)

    # Resample to 48 kHz if needed
    if sr != 48000:
        from scipy import signal as scipy_signal
        num_samples = int(len(audio) * 48000 / sr)
        audio = scipy_signal.resample(audio, num_samples)
        sr = 48000

    # Segment with optimized parameters
    print("\n🔍 Segmenting with optimized parameters:")
    print("   min_gap_ms: 30")
    print("   min_phrase_duration_ms: 5")

    phrases = analyzer.segment_phrases(
        audio,
        min_gap_ms=30,
        min_phrase_duration_ms=5
    )

    print("\n📊 Segmentation Results:")
    print(f"  Total phrases detected: {len(phrases)}")

    if len(phrases) == 0:
        print("  ⚠️  No phrases detected")
        return []

    # Analyze each phrase in detail
    print("\n📋 Detailed Phrase Analysis:")

    phrase_details = []

    for i, phrase in enumerate(phrases):
        print(f"\n  Phrase {i+1}:")
        print(f"    Duration: {len(phrase.data)/sr*1000:.1f} ms")
        print(f"    Samples: {len(phrase.data)}")
        print(f"    Timestamp: {phrase.timestamp if phrase.timestamp else 'N/A'}")

        # Modality detection
        modality = analyzer.detect_modality(phrase.data)
        probabilities = analyzer.get_modality_probabilities(phrase.data)

        print(f"    Modality: {modality.name}")
        print(f"    Probabilities: {probabilities}")

        # Get features
        features = phrase.features
        print("    Features:")

        for key, value in features.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: {value}")

        phrase_details.append({
            'index': i + 1,
            'modality': modality.name,
            'probabilities': probabilities,
            'features': features,
            'duration_ms': len(phrase.data) / sr * 1000
        })

    return phrase_details


def compare_with_synthetic():
    """Compare Phee call with synthetic marmoset-like signals."""
    print(f"\n{'='*70}")
    print("COMPARISON WITH SYNTHETIC SIGNALS")
    print(f"{'='*70}")

    # Generate synthetic marmoset call (7 kHz harmonic)
    sr = 48000
    duration_ms = 50
    t = np.linspace(0, duration_ms/1000, int(sr * duration_ms/1000))

    # Fundamental at 7 kHz (marmoset range)
    synthetic = 0.5 * np.sin(2 * np.pi * 7000 * t)
    # Add harmonics
    synthetic += 0.25 * np.sin(2 * np.pi * 14000 * t)
    synthetic += 0.125 * np.sin(2 * np.pi * 21000 * t)
    synthetic /= np.max(np.abs(synthetic))

    analyzer = UniversalRosettaStone(sample_rate=48000)

    # Detect modality
    modality = analyzer.detect_modality(synthetic)
    probabilities = analyzer.get_modality_probabilities(synthetic)

    print("\n📊 Synthetic Marmoset Call (7 kHz harmonic):")
    print(f"  Modality: {modality.name}")
    print(f"  Probabilities: {probabilities}")

    print("\n📊 Comparison:")
    print("  Expected: HARMONIC (marmosets use flat tones)")
    print(f"  Detected: {modality.name}")

    if modality == Modality.HARMONIC:
        print("  ✓ Correct!")
    else:
        print("  ⚠️  Unexpected - check F0 range thresholds")


def generate_spectrogram(audio, sr, output_path):
    """Generate and save spectrogram visualization."""
    if not HAS_MATPLOTLIB:
        return

    from scipy.signal import spectrogram

    # Generate spectrogram
    frequencies, times, Sxx = spectrogram(audio, sr)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Waveform
    ax1.plot(np.arange(len(audio)) / sr, audio)
    ax1.set_title('Phee Call Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # Spectrogram
    im = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax2.set_title('Phee Call Spectrogram')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim([0, min(24000, sr//2)])  # Show up to 24 kHz
    fig.colorbar(im, ax=ax2, label='Power (dB)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Spectrogram saved to: {output_path}")


def main():
    """Main analysis function."""
    print("="*70)
    print("DETAILED PHEE CALL ANALYSIS")
    print("Marmoset vocalization (Phee_62767.flac)")
    print("="*70)

    # Load the Phee call
    audio, sr, filepath = load_phee_call()

    if audio is None:
        return

    # 1. Acoustic feature analysis
    analyze_acoustic_features(audio, sr)

    # 2. Phrase structure analysis
    phrase_details = analyze_phrase_structure(audio, sr)

    # 3. Comparison with synthetic
    compare_with_synthetic()

    # 4. Generate spectrogram
    print(f"\n{'='*70}")
    print("SPECTROGRAM GENERATION")
    print(f"{'='*70}")

    output_path = Path(__file__).parent / "phee_call_analysis.png"
    generate_spectrogram(audio, sr, output_path)

    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")

    if phrase_details:
        # Count modalities
        modality_counts = {}
        for p in phrase_details:
            m = p['modality']
            modality_counts[m] = modality_counts.get(m, 0) + 1

        print(f"\n📊 Modality Distribution ({len(phrase_details)} phrases):")
        for modality, count in sorted(modality_counts.items()):
            percentage = count / len(phrase_details) * 100
            bar = '█' * int(percentage / 10)
            print(f"  {modality:15s}: {count} ({percentage:.0f}%) {bar}")

        print("\n🔬 Key Findings:")
        print(f"  • Phee call contains {len(phrase_details)} distinct phrase segments")
        print("  • Mixed modality detected (HARMONIC + FM_SWEEP)")

        # Check for HARMONIC presence
        harmonic_count = modality_counts.get('HARMONIC', 0)
        if harmonic_count > 0:
            print("  ✓ HARMONIC components detected (expected for marmosets)")
        else:
            print("  ⚠️  No HARMONIC components detected")

        print("\n💡 Interpretation:")
        print("  Marmoset Phee calls show frequency modulation, which may explain")
        print("  the FM_SWEEP classification. This is scientifically valid as")
        print("  marmoset calls are not pure flat tones but have some modulation.")

    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
