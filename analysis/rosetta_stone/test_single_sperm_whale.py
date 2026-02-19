#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone

# Load first 10 seconds
filepath = Path.home() / "birdsong_analysis/data/Dominica_dataset/Signal_parts/SW_1_filtered.wav"
print(f"Loading: {filepath.name}")

# Get sample rate first
info = sf.info(filepath)
sr = info.samplerate
audio, sr = sf.read(filepath, start=0, stop=int(10 * sr))
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

print(f"Loaded: {len(audio)} samples at {sr} Hz")
print(f"Duration: {len(audio) / sr * 1000:.0f} ms")
print(f"RMS: {np.sqrt(np.mean(audio**2)):.6f}")

# Frequency analysis
from scipy.fft import fft, fftfreq

fft_result = fft(audio)
freqs = fftfreq(len(audio), 1 / sr)
magnitude = np.abs(fft_result)

pos_freqs = freqs[: len(freqs) // 2]
pos_magnitude = magnitude[: len(magnitude) // 2]

# Energy bands
bands = [
    ("0-2 kHz", 0, 2000),
    ("2-8 kHz (SW clicks)", 2000, 8000),
    ("8-15 kHz", 8000, 15000),
    (">15 kHz", 15000, sr // 2),
]

print("\n📊 Energy Distribution:")
total = np.sum(pos_magnitude**2)
for name, low, high in bands:
    mask = (pos_freqs >= low) & (pos_freqs < high)
    e = np.sum(pos_magnitude[mask] ** 2) / total * 100
    print(f"  {name:20s}: {e:5.1f}%")

# Try segmentation
print("\n🔍 Segmentation:")
analyzer = UniversalRosettaStone(sample_rate=sr)

for gap in [20, 50, 100, 200]:
    for dur in [10, 20, 50]:
        try:
            phrases = analyzer.segment_phrases(audio, min_gap_ms=gap, min_phrase_duration_ms=dur)
            if len(phrases) > 0:
                print(f"  gap={gap:3d}ms, dur={dur:2d}ms: {len(phrases)} phrases ✓")
        except Exception:
            pass

# Try with best params
phrases = analyzer.segment_phrases(audio, min_gap_ms=50, min_phrase_duration_ms=10)
print(f"\n📊 Result: {len(phrases)} phrases")

if len(phrases) > 0:
    for i in range(min(5, len(phrases))):
        m = analyzer.detect_modality(phrases[i].data)
        probs = analyzer.get_modality_probabilities(phrases[i].data)
        print(f"  Phrase {i + 1}: {m.name} - {probs}")
else:
    # Try whole file
    m = analyzer.detect_modality(audio)
    probs = analyzer.get_modality_probabilities(audio)
    print(f"  Whole file: {m.name} - {probs}")

print("\n✅ Test complete!")
