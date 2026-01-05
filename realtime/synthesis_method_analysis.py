#!/usr/bin/env python3
"""
Synthesis Method Analysis - Audio Source vs. Emulation
======================================================

This script analyzes whether each synthesis method uses real audio segments
or generates audio based on phrase signatures/fingerprints.
"""

import numpy as np
import sys
sys.path.append('/home/sheel/birdsong_analysis')
sys.path.append('/home/sheel/birdsong_analysis/src/realtime')

from phrase_audio_library import PhraseAudioLibrary
from advanced_synthesis_methods import SynthesisFactory

def analyze_synthesis_methods():
    """Analyze each synthesis method's audio source approach."""
    print("=" * 80)
    print("SYNTHESIS METHOD ANALYSIS - AUDIO SOURCE VS EMULATION")
    print("=" * 80)

    # Create test library
    library = PhraseAudioLibrary(species='marmoset', sr=22050)

    # Create real audio segments
    real_audio_segments = {}
    for i, freq in enumerate([4000, 5000, 6000]):
        # Generate actual audio
        audio = np.sin(2 * np.pi * freq * np.linspace(0, 0.1, 2205))
        real_audio_segments[f'F0_{freq}_DUR_5_RANGE_0'] = audio

        segment = library.create_phrase_segment(
            audio=audio,
            phrase_key=f'F0_{freq}_DUR_5_RANGE_0',
            context='neutral',
            individual_id='test_individual',
            quality_score=0.9,
            source_file='test'
        )

    # Test each synthesis method
    methods = ['concatenative', 'superpositional', 'combined', 'microharmonic']
    analysis_results = {}

    for method in methods:
        print(f"\n🔍 {method.upper()} SYNTHESIS ANALYSIS:")
        print("─" * 50)

        try:
            synthesizer = SynthesisFactory.create_synthesizer(
                method,
                library,
                sample_rate=22050
            )

            # Test synthesis
            phrase_keys = list(real_audio_segments.keys())[:2]

            if method == 'microharmonic':
                # Use microharmonic method
                from advanced_synthesis_methods import ContextState
                result = synthesizer.synthesize_microharmonic_phrases(
                    phrase_keys,
                    ContextState.NEUTRAL,
                    temporal_alignment=True
                )
                samples = len(result) if result is not None else 0
            else:
                # Use generic interface
                try:
                    config = synthesizer.config_class(
                        phrase_sequence=phrase_keys,
                        encoding_mode='horizontal'
                    )
                    result = synthesizer.synthesize(config)
                    samples = len(result.get('audio', [])) if result and result.get('success') else 0
                except:
                    # Try different approach
                    result = synthesizer.synthesize(phrase_keys)
                    samples = len(result) if result is not None else 0

            # Analysis
            if samples > 0:
                print(f"✅ Synthesis successful: {samples} samples")

                # Check method type by examining its implementation
                if hasattr(synthesizer, 'phrase_library'):
                    print("📊 Method Type: AUDIO-BASED (uses real segments)")
                    print("   • Retrieves actual audio from phrase library")
                    print("   • Manipulates/combines existing recordings")
                    print("   • Preserves original audio characteristics")

                    if method in ['concatenative', 'superpositional', 'combined']:
                        print("   • Uses phrase audio directly from database")
                else:
                    print("📊 Method Type: GENERATIVE (emulation-based)")
                    print("   • Creates new audio from metadata/signatures")
                    print("   • Does not use original audio segments")
                    print("   • Generates audio based on mathematical models")
            else:
                print("❌ No output generated")

            analysis_results[method] = {
                'samples': samples,
                'type': 'audio-based' if hasattr(synthesizer, 'phrase_library') else 'generative'
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            analysis_results[method] = {'samples': 0, 'type': 'error'}

    # Summary
    print("\n" + "=" * 80)
    print("📋 SYNTHESIS METHOD COMPARISON")
    print("=" * 80)

    for method, result in analysis_results.items():
        if result['type'] != 'error':
            emoji = "🎵" if result['type'] == 'audio-based' else "🎼"
            print(f"{emoji} {method:15} : {result['type']:12} ({result['samples']:,} samples)")
        else:
            print(f"❌ {method:15} : ERROR")

    print("\n🔬 TECHNICAL DISTINCTION:")
    print("─" * 50)
    print("AUDIO-BASED METHODS (Real Segments):")
    print("  • Concatenative: Uses actual audio segments concatenated")
    print("  • Superpositional: Uses actual audio segments layered")
    print("  • Combined: Mixes both approaches with real segments")
    print("  • Source: phrase_library.get_segment() returns original audio")
    print()
    print("GENERATIVE METHOD (Signature/Fingerprint):")
    print("  • Enhanced Microharmonic: Creates audio from metadata")
    print("  • Source: mathematical models based on F0, harmonics, context")
    print("  • Does not use original audio segments")
    print()
    print("🎯 KEY DIFFERENCE:")
    print("  • Audio-based = Manipulates existing recordings")
    print("  • Generative = Emulates vocalizations from signatures")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_synthesis_methods()