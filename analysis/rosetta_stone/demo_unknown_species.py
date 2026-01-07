#!/usr/bin/env python3
"""
Demo: Universal Rosetta Stone - Unknown Species Analysis

This demo demonstrates how the Universal Rosetta Stone system can analyze
vocalizations from unknown species without prior species-specific knowledge.

The system:
1. Detects acoustic modalities (harmonic, FM sweep, transient, rhythmic)
2. Segments audio into individual phrases
3. Clusters similar phrases into vocabulary
4. Discovers grammatical rules and sentence structure
5. Synthesizes novel sequences for interaction

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import os
import sys

import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

from universal_rosetta_stone import Modality, UniversalRosettaStone
from universal_synthesizer import UniversalSynthesizer


def generate_unknown_species_audio():
    """
    Generate synthetic audio representing an "unknown species"
    with mixed modalities and complex patterns.
    """
    sample_rate = 48000
    duration = 0.05
    gap_duration = 0.01
    gap_samples = int(gap_duration * sample_rate)

    print("🎵 Generating unknown species audio...")

    # Simulate a complex vocalization pattern
    # Sequence: Harmonic -> FM Sweep -> Transient -> Harmonic -> FM Sweep

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Phrase 1: Harmonic tone (4kHz)
    phrase1 = np.sin(2 * np.pi * 4000 * t)

    # Phrase 2: FM sweep (5kHz -> 7kHz)
    phrase2_t = np.linspace(0, duration, int(sample_rate * duration))
    phrase2_freq = 5000 + 2000 * phrase2_t / duration
    phrase2 = np.sin(2 * np.pi * np.cumsum(phrase2_freq) / sample_rate)

    # Phrase 3: Transient click (high frequency)
    phrase3 = np.zeros(int(sample_rate * duration))
    phrase3[len(phrase3)//2-50:len(phrase3)//2+50] = np.sin(2 * np.pi * 12000 * t[:100])
    phrase3 *= np.exp(-np.linspace(0, 10, len(phrase3)))

    # Phrase 4: Harmonic tone (6kHz) - similar to phrase1 but different pitch
    phrase4 = np.sin(2 * np.pi * 6000 * t)

    # Phrase 5: FM sweep (3kHz -> 5kHz) - different sweep
    phrase5_t = np.linspace(0, duration, int(sample_rate * duration))
    phrase5_freq = 3000 + 2000 * phrase5_t / duration
    phrase5 = np.sin(2 * np.pi * np.cumsum(phrase5_freq) / sample_rate)

    # Combine with gaps
    audio_parts = [phrase1, np.zeros(gap_samples),
                  phrase2, np.zeros(gap_samples),
                  phrase3, np.zeros(gap_samples),
                  phrase4, np.zeros(gap_samples),
                  phrase5]

    audio = np.concatenate(audio_parts)

    print(f"   Generated {len(audio)/sample_rate:.2f}s of audio with {len(audio_parts)} phrases")
    return audio


def analyze_unknown_species(audio):
    """
    Analyze unknown species audio using Universal Rosetta Stone.
    """
    print("\n🔍 Analyzing unknown species vocalizations...")

    # Initialize analyzer
    analyzer = UniversalRosettaStone(sample_rate=48000)

    start_time = time.time()

    # Discover vocabulary and grammar
    vocabulary, grammar = analyzer.discover_grammar(audio)

    analysis_time = time.time() - start_time

    print(f"   Analysis completed in {analysis_time:.2f}s")
    print(f"   Discovered {len(vocabulary)} unique phrases")
    print(f"   Discovered {len(grammar)} syntactic rules")

    return analyzer, vocabulary, grammar


def display_analysis_results(analyzer, vocabulary, grammar):
    """Display detailed analysis results."""
    print("\n📊 Analysis Results:")
    print("=" * 50)

    # Phrase details
    print("\nDiscovered Phrases:")
    for cluster_id, phrase_sig in vocabulary.items():
        print(f"  Phrase {cluster_id}:")
        print(f"    Modality: {phrase_sig.modality.name}")
        print(f"    Duration: {phrase_sig.features['duration_ms']:.1f}ms")

        if phrase_sig.modality == Modality.HARMONIC:
            print(f"    F0 Mean: {phrase_sig.features['f0_mean']:.0f}Hz")
            print(f"    F0 Std: {phrase_sig.features['f0_std']:.0f}Hz")
        elif phrase_sig.modality == Modality.FM_SWEEP:
            print(f"    Frequency Range: {phrase_sig.features['start_freq']:.0f}Hz → {phrase_sig.features['end_freq']:.0f}Hz")
            print(f"    Sweep Rate: {phrase_sig.features['freq_slope']:.0f}Hz/s")
        elif phrase_sig.modality == Modality.TRANSIENT:
            print(f"    Spectral Centroid: {phrase_sig.features['spectral_centroid']:.0f}Hz")
            print(f"    RMS Energy: {phrase_sig.features['rms']:.3f}")

    # Grammar rules
    print("\nDiscovered Grammar Rules:")
    if grammar:
        # Sort by frequency
        sorted_rules = sorted(grammar.items(), key=lambda x: x[1], reverse=True)
        for (from_phrase, to_phrase), count in sorted_rules:
            print(f"  Phrase {from_phrase} → Phrase {to_phrase}: {count} occurrences")
    else:
        print("  No clear grammatical patterns discovered")

    # Statistics
    stats = analyzer.get_phrase_statistics()
    print("\nSystem Statistics:")
    print(f"  Total Phrases: {stats['total_phrases']}")

    if stats['modality_distribution']:
        print("  Modality Distribution:")
        for modality, count in stats['modality_distribution'].items():
            print(f"    {modality}: {count} phrases")


def synthesize_interactive_response(analyzer, vocabulary, grammar):
    """
    Synthesize an interactive response based on discovered patterns.
    """
    print("\n🎼 Synthesizing interactive response...")

    # Create synthesizer
    synthesizer = UniversalSynthesizer(vocabulary, grammar)

    # Generate a sequence of phrases
    sequence = synthesizer.generate_sequence(num_phrases=4)
    print(f"   Generated sequence: {sequence}")

    # Synthesize audio
    synthesized_audio = synthesizer.synthesize_audio(
        sequence,
        phrase_duration_ms=50,
        gap_ms=10,
        sample_rate=48000
    )

    print(f"   Synthesized {len(synthesized_audio)/48000:.2f}s of audio")

    return synthesizer, synthesized_audio, sequence


def demonstrate_cross_species_learning():
    """
    Demonstrate how the system can learn from multiple species.
    """
    print("\n🌐 Cross-Species Learning Demo:")
    print("-" * 30)

    # Simulate different species with different characteristics
    species_data = {
        'Marmoset-like': {
            'type': 'harmonic',
            'frequencies': [4000, 5000, 6000],
            'pattern': 'A->B->C'
        },
        'Bat-like': {
            'type': 'fm_sweep',
            'frequencies': [(20000, 30000), (25000, 28000)],
            'pattern': 'X->Y->X'
        },
        'Whale-like': {
            'type': 'transient',
            'frequencies': [1000, 1500],
            'pattern': 'P->Q->P'
        }
    }

    # Analyze each species
    analyzer = UniversalRosettaStone(sample_rate=48000)

    for species_name, species_info in species_data.items():
        print(f"\nAnalyzing {species_name} vocalizations...")

        # Generate species-specific audio
        sample_rate = 48000
        duration = 0.05
        gap_samples = int(0.01 * sample_rate)

        phrases = []

        if species_info['type'] == 'harmonic':
            for freq in species_info['frequencies']:
                t = np.linspace(0, duration, int(sample_rate * duration))
                phrases.append(np.sin(2 * np.pi * freq * t))

        elif species_info['type'] == 'fm_sweep':
            for start_freq, end_freq in species_info['frequencies']:
                t = np.linspace(0, duration, int(sample_rate * duration))
                instantaneous_freq = start_freq + (end_freq - start_freq) * t / duration
                phrase = np.sin(2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate)
                phrases.append(phrase)

        elif species_info['type'] == 'transient':
            for freq in species_info['frequencies']:
                t = np.linspace(0, duration, int(sample_rate * duration))
                phrase = np.zeros_like(t)
                phrase[len(phrase)//2-50:len(phrase)//2+50] = np.sin(2 * np.pi * freq * t[:100])
                phrase *= np.exp(-np.linspace(0, 10, len(phrase)))
                phrases.append(phrase)

        # Combine phrases
        audio_parts = []
        for phrase in phrases:
            audio_parts.append(phrase)
            audio_parts.append(np.zeros(gap_samples))

        audio = np.concatenate(audio_parts[:-1])  # Remove last gap

        # Analyze
        vocabulary, grammar = analyzer.discover_grammar(audio)
        print(f"   Discovered {len(vocabulary)} phrases and {len(grammar)} rules")

        # Determine primary modality
        modalities = [v.modality for v in vocabulary.values()]
        if modalities:
            primary_modality = max(set(modalities), key=modalities.count)
            print(f"   Primary modality: {primary_modality.name}")


def main():
    """Main demo function."""
    print("🌟 Universal Rosetta Stone - Unknown Species Analysis")
    print("=" * 60)
    print("\nThis demo demonstrates a physics-based approach to analyzing")
    print("animal vocalizations from unknown species without prior knowledge.")

    # Generate test audio
    audio = generate_unknown_species_audio()

    # Analyze the audio
    analyzer, vocabulary, grammar = analyze_unknown_species(audio)

    # Display results
    display_analysis_results(analyzer, vocabulary, grammar)

    # Synthesize response
    synthesizer, synthesized_audio, sequence = synthesize_interactive_response(
        analyzer, vocabulary, grammar
    )

    # Show synthesizer statistics
    synth_stats = synthesizer.get_statistics()
    print("\nSynthesizer Statistics:")
    print(f"  Vocabulary size: {synth_stats['vocabulary_size']}")
    print(f"  Grammar rules: {synth_stats['grammar_rules']}")

    # Cross-species learning demo
    demonstrate_cross_species_learning()

    # Final summary
    print("\n" + "=" * 60)
    print("🎯 Universal Rosetta Stone Key Achievements:")
    print("  ✓ Physics-based modality detection")
    print("  ✓ Species-agnostic vocabulary building")
    print("  ✓ Grammar discovery from sequences")
    print("  ✓ Novel sequence synthesis")
    print("  ✓ Cross-species pattern recognition")

    print("\n🚀 The system can now analyze any unknown acoustic signal")
    print("   and generate novel responses based on discovered patterns!")

    # Save synthesized audio (optional)
    try:
        import soundfile as sf
        output_file = "unknown_species_response.wav"
        sf.write(output_file, synthesized_audio, 48000)
        print(f"\n💾 Saved synthesized response to: {output_file}")
    except ImportError:
        print("\n💡 Install soundfile to save audio: pip install soundfile")


if __name__ == "__main__":
    main()
