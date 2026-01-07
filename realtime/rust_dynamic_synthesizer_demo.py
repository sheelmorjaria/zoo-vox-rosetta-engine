#!/usr/bin/env python3
"""
Rust Dynamic Microharmonic Synthesizer Demo
==========================================

This script demonstrates the Rust Dynamic Microharmonic Synthesizer
via PyO3 bindings, showing:
1. Single phrase synthesis with micro-dynamics
2. Sequence synthesis (sentences)
3. Parameter generation for different species

Usage:
    python rust_dynamic_synthesizer_demo.py

Requirements:
    - Rust library compiled with python-bindings feature
    - PyO3 installed in Python environment
"""

import json
import sys
from pathlib import Path

import soundfile as sf

# Import Rust synthesizer via PyO3
try:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "technical_architecture",
        "/mnt/c/Users/sheel/Desktop/src/technical_architecture/target/release/libtechnical_architecture.so",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["technical_architecture"] = module
    spec.loader.exec_module(module)
    DynamicMicroharmonicSynthesizer = module.DynamicMicroharmonicSynthesizer
except ImportError as e:
    print(f"❌ Failed to import Rust synthesizer: {e}")
    print("\nMake sure the Rust library is compiled with:")
    print("  cargo build --features python-bindings --release")
    print("\nThe compiled library should be in the Python path.")
    sys.exit(1)


def demo_single_phrase():
    """Demonstrate single phrase synthesis."""
    print("\n" + "=" * 80)
    print("DEMO 1: SINGLE PHRASE SYNTHESIS")
    print("=" * 80)

    # Create synthesizer
    synthesizer = DynamicMicroharmonicSynthesizer(sample_rate=22050)

    # Marmoset-style phrase
    print("\n🎵 Synthesizing marmoset phrase...")
    audio_marmoset = synthesizer.synthesize_phrase(
        f0_base=8000.0,
        duration_ms=100.0,
        attack_ms=10.0,
        decay_ms=30.0,
        sustain_level=0.7,
        vibrato_rate_hz=7.0,
        vibrato_depth_cents=25.0,
        jitter_amount=0.025,
    )

    print(
        f"   Generated {len(audio_marmoset)} samples ({len(audio_marmoset) / 22050 * 1000:.1f} ms)"
    )

    # Bat-style phrase
    print("\n🎵 Synthesizing bat phrase...")
    audio_bat = synthesizer.synthesize_phrase(
        f0_base=24000.0,
        duration_ms=75.0,
        attack_ms=10.0,
        decay_ms=28.0,
        sustain_level=0.7,
        vibrato_rate_hz=7.5,
        vibrato_depth_cents=25.0,
        jitter_amount=0.025,
    )

    print(f"   Generated {len(audio_bat)} samples ({len(audio_bat) / 22050 * 1000:.1f} ms)")

    # Save to files
    output_dir = Path("/home/sheel/birdsong_analysis/src/validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    sf.write(str(output_dir / "marmoset_phrase_rust.wav"), audio_marmoset, 22050)
    sf.write(str(output_dir / "bat_phrase_rust.wav"), audio_bat, 22050)

    print(f"\n💾 Saved to {output_dir}")

    return audio_marmoset, audio_bat


def demo_default_parameters():
    """Demonstrate species-specific default parameters."""
    print("\n" + "=" * 80)
    print("DEMO 2: SPECIES-SPECIFIC DEFAULT PARAMETERS")
    print("=" * 80)

    synthesizer = DynamicMicroharmonicSynthesizer(sample_rate=22050)

    # Get marmoset defaults
    print("\n📊 Marmoset default parameters (F0=8kHz, Duration=100ms):")
    marmoset_params_json = synthesizer.marmoset_default(8000.0, 100.0)
    marmoset_params = json.loads(marmoset_params_json)

    print(json.dumps(marmoset_params, indent=2))

    # Get bat defaults
    print("\n📊 Bat default parameters (F0=24kHz, Duration=75ms):")
    bat_params_json = synthesizer.bat_default(24000.0, 75.0)
    bat_params = json.loads(bat_params_json)

    print(json.dumps(bat_params, indent=2))


def demo_sequence_synthesis():
    """Demonstrate sequence synthesis (sentences)."""
    print("\n" + "=" * 80)
    print("DEMO 3: SEQUENCE SYNTHESIS (SENTENCES)")
    print("=" * 80)

    synthesizer = DynamicMicroharmonicSynthesizer(sample_rate=22050)

    # Create a 3-phrase sequence simulating ascending syntax
    phrase1 = {
        "f0_base": 7000.0,
        "duration_ms": 80.0,
        "attack_ms": 10.0,
        "decay_ms": 25.0,
        "sustain_level": 0.7,
        "vibrato_rate_hz": 7.0,
        "vibrato_depth_cents": 25.0,
        "jitter_amount": 0.02,
        "shimmer_amount": 0.01,
        "spectral_tilt": -6.0,
        "hnr_db": 20.0,
    }

    phrase2 = {
        "f0_base": 8500.0,
        "duration_ms": 80.0,
        "attack_ms": 10.0,
        "decay_ms": 25.0,
        "sustain_level": 0.7,
        "vibrato_rate_hz": 7.5,
        "vibrato_depth_cents": 25.0,
        "jitter_amount": 0.02,
        "shimmer_amount": 0.01,
        "spectral_tilt": -6.0,
        "hnr_db": 20.0,
    }

    phrase3 = {
        "f0_base": 10000.0,
        "duration_ms": 80.0,
        "attack_ms": 10.0,
        "decay_ms": 25.0,
        "sustain_level": 0.7,
        "vibrato_rate_hz": 8.0,
        "vibrato_depth_cents": 25.0,
        "jitter_amount": 0.02,
        "shimmer_amount": 0.01,
        "spectral_tilt": -6.0,
        "hnr_db": 20.0,
    }

    # Combine into sequence
    phrase_sequence_json = json.dumps([phrase1, phrase2, phrase3])

    print("\n🎵 Synthesizing ascending sequence (7kHz → 8.5kHz → 10kHz)...")
    audio_sequence = synthesizer.synthesize_sequence(
        phrase_params_json=phrase_sequence_json, crossfade_ms=10.0
    )

    print(
        f"   Generated {len(audio_sequence)} samples ({len(audio_sequence) / 22050 * 1000:.1f} ms)"
    )

    # Save to file
    output_dir = Path("/home/sheel/birdsong_analysis/src/validation_results")
    sf.write(str(output_dir / "ascending_sequence_rust.wav"), audio_sequence, 22050)

    print(f"💾 Saved to {output_dir / 'ascending_sequence_rust.wav'}")

    return audio_sequence


def demo_random_parameters():
    """Demonstrate random parameter generation."""
    print("\n" + "=" * 80)
    print("DEMO 4: RANDOM PARAMETER GENERATION")
    print("=" * 80)

    synthesizer = DynamicMicroharmonicSynthesizer(sample_rate=22050)

    # Generate random parameters
    print("\n🎲 Generating random parameters (F0=8kHz, Duration=100ms, variability=0.5)...")
    random_params_json = synthesizer.generate_random_params(8000.0, 100.0, 0.5)
    random_params = json.loads(random_params_json)

    print("\nRandom parameters:")
    print(json.dumps(random_params, indent=2))


def main():
    """Main demo function."""
    print("=" * 80)
    print("RUST DYNAMIC MICROHARMONIC SYNTHESIZER DEMO")
    print("=" * 80)
    print("\n✅ Successfully imported DynamicMicroharmonicSynthesizer from Rust")
    print("   via PyO3 bindings")

    # Run demos
    try:
        demo_single_phrase()
        demo_default_parameters()
        demo_sequence_synthesis()
        demo_random_parameters()

        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETE!")
        print("=" * 80)

        print(
            "\n📂 Generated files saved to: /home/sheel/birdsong_analysis/src/validation_results/"
        )
        print("   - marmoset_phrase_rust.wav")
        print("   - bat_phrase_rust.wav")
        print("   - ascending_sequence_rust.wav")

        print("\n🎯 NEXT STEPS:")
        print("   1. Extract real micro-dynamics from audio library")
        print("   2. Run validation with Rust-synthesized audio")
        print("   3. Compare t-SNE congruence scores")

        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
