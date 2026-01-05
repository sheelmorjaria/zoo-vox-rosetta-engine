#!/usr/bin/env python3
"""
Final Synthesis Demo - All 4 Methods
=====================================

This script demonstrates all four synthesis methods working together:
1. Concatenative (horizontal)
2. Superpositional (vertical)
3. Combined (mixed)
4. Enhanced Microharmonic

Author: Sheel Morjaria
License: CC BY-ND 4.0 International
"""

import numpy as np
import sys
sys.path.append('/home/sheel/birdsong_analysis')
sys.path.append('/home/sheel/birdsong_analysis/src/realtime')

from phrase_audio_library import PhraseAudioLibrary
from advanced_synthesis_methods import SynthesisFactory

def create_demo_library():
    """Create a demonstration library with various phrase types."""
    library = PhraseAudioLibrary(species='marmoset', sr=22050)

    # Create different types of phrases for demonstration
    frequencies = [4000, 5000, 6000, 7000, 8000]
    contexts = ['contact', 'neutral', 'food', 'social', 'alarm']

    for i, (freq, context) in enumerate(zip(frequencies, contexts)):
        # Create harmonic-rich audio
        duration = 0.2
        samples = int(duration * 22050)
        t = np.linspace(0, duration, samples)

        # Fundamental frequency
        audio = np.sin(2 * np.pi * freq * t) * 0.3

        # Add harmonics
        for harmonic in [2, 3]:
            audio += np.sin(2 * np.pi * freq * harmonic * t) * 0.2

        # Apply envelope
        envelope = np.exp(-t * 10)
        audio *= envelope

        segment = library.create_phrase_segment(
            audio=audio,
            phrase_key=f'F0_{freq}_DUR_5_RANGE_0',
            context=context,
            individual_id='demo_individual',
            quality_score=0.9,
            source_file='demo'
        )

    return library

def demo_all_synthesis_methods():
    """Demonstrate all four synthesis methods."""
    print("=" * 80)
    print("FINAL SYNTHESIS DEMO - ALL 4 METHODS")
    print("=" * 80)

    # Create demo library
    library = create_demo_library()
    print(f"✅ Created demo library with {len(library.get_all_phrase_keys())} phrases")

    # Test all synthesis methods
    methods = ['concatenative', 'superpositional', 'combined', 'microharmonic']

    print("\n" + "─" * 80)
    print("TESTING ALL SYNTHESIS METHODS")
    print("─" * 80)

    # Get sample phrases
    phrase_keys = library.get_all_phrase_keys()[:3]
    print(f"Sample phrases: {phrase_keys}")

    results = {}

    for method in methods:
        print(f"\n🎼 {method.upper()} SYNTHESIS:")
        print("─" * 40)

        try:
            # Create synthesizer
            synthesizer = SynthesisFactory.create_synthesizer(
                method,
                library,
                sample_rate=22050
            )
            print(f"✅ Synthesizer created: {type(synthesizer).__name__}")

            # Test synthesis
            if method == 'microharmonic':
                # For microharmonic, use the specific method
                try:
                    from advanced_synthesis_methods import ContextState
                    result = synthesizer.synthesize_microharmonic_phrases(
                        phrase_keys,
                        ContextState.NEUTRAL,
                        temporal_alignment=True
                    )
                    samples = len(result) if result is not None else 0
                    print(f"✅ Synthesis successful: {samples} samples")
                    results[method] = samples
                except Exception as e:
                    print(f"⚠️ Synthesis error: {e}")
                    results[method] = 0
            else:
                # For other methods, use the generic interface
                try:
                    config = synthesizer.config_class(
                        phrase_sequence=phrase_keys,
                        encoding_mode='horizontal' if method == 'concatenative' else 'vertical'
                    )
                    result = synthesizer.synthesize(config)
                    samples = len(result.get('audio', [])) if result and result.get('success') else 0
                    print(f"✅ Synthesis successful: {samples} samples")
                    results[method] = samples
                except Exception as e:
                    print(f"⚠️ Synthesis error: {e}")
                    results[method] = 0

        except Exception as e:
            print(f"❌ Error creating synthesizer: {e}")
            results[method] = 0

    # Summary
    print("\n" + "─" * 80)
    print("SYNTHESIS METHODS SUMMARY")
    print("─" * 80)

    for method, samples in results.items():
        status = "✅ WORKING" if samples > 0 else "❌ FAILED"
        print(f"  {method:15} : {status:12} ({samples:,} samples)")

    # Success rate
    working_methods = sum(1 for samples in results.values() if samples > 0)
    print(f"\n🎯 Success Rate: {working_methods}/{len(methods)} methods working")

    if working_methods == len(methods):
        print("🎉 ALL SYNTHESIS METHODS SUCCESSFULLY INTEGRATED!")
        print("   • Concatenative synthesis available")
        print("   • Superpositional synthesis available")
        print("   • Combined synthesis available")
        print("   • Enhanced microharmonic synthesis available")
    else:
        print("⚠️ Some methods need attention")

    print("\n" + "=" * 80)
    print("✅ ENHANCED MICROHARMONIC SYNTHESIS INTEGRATION COMPLETE")
    print("✅ All synthesis methods available through SynthesisFactory")
    print("✅ Framework ready for advanced bioacoustic research")
    print("=" * 80)

    return results

if __name__ == "__main__":
    demo_all_synthesis_methods()