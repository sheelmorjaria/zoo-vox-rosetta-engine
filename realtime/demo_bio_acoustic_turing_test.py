#!/usr/bin/env python3
"""
Bio-Acoustic Turing Test - Demo
================================

Demonstrates the complete Bio-Acoustic Turing Test framework for
validating that animals cannot distinguish between natural and
granular-synthesized vocalizations.

Usage:
    python demo_bio_acoustic_turing_test.py

This demo shows:
1. Loading natural (concatenative) stimuli
2. Generating granular-synthesized stimuli
3. Running simulated trials
4. Statistical analysis and hypothesis evaluation
"""

import json
import numpy as np
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from bio_acoustic_turing_test import (
    BioAcousticTuringTest,
    StimulusController,
    ResponseRecorder,
    ExperimentDesign,
    StatisticalAnalyzer
)

# Import granular synthesizer
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'technical_architecture',
        '/mnt/c/Users/sheel/Desktop/src/technical_architecture/target/release/libtechnical_architecture.so'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['technical_architecture'] = module
    spec.loader.exec_module(module)
    GranularConcatenativeSynthesizer = module.GranularConcatenativeSynthesizer
    print("✅ Successfully imported GranularConcatenativeSynthesizer from Rust")
except ImportError as e:
    print(f"❌ Could not import Rust synthesizer: {e}")
    print("Falling back to Python-only demo")
    GranularConcatenativeSynthesizer = None


def create_simulated_phee_call(sample_rate: int, f0: float, duration_ms: float) -> np.ndarray:
    """Create a simulated phee call (marmoset vocalization)."""
    duration_sec = duration_ms / 1000.0
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples)

    # Phee call: harmonic stack with vibrato
    audio = np.zeros(num_samples)
    for harmonic in range(1, 4):  # First 3 harmonics
        amplitude = 0.5 / harmonic
        vibrato_rate = 15.0  # Hz
        vibrato_depth = 0.02  # Fraction of F0

        # Vibrato
        vibrato_osc = np.sin(2 * np.pi * vibrato_rate * t)
        vibrato_ratio = 1.0 + vibrato_depth * vibrato_osc

        # Harmonic with vibrato
        phase = 2 * np.pi * f0 * harmonic * vibrato_ratio * t
        audio += amplitude * np.sin(phase)

    # Apply envelope
    attack_samples = int(0.01 * sample_rate)  # 10ms attack
    decay_samples = int(0.03 * sample_rate)  # 30ms decay
    envelope = np.ones(num_samples)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

    return audio * envelope * 0.5


def demo_concatenative_baseline():
    """
    Demo Phase 1: Concatenative Baseline

    Establish baseline response rate using natural recordings.
    This represents the "gold standard" for bio-acoustic validity.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: CONCATENATIVE BASELINE (Natural Recordings)")
    print("=" * 80)

    # Create test instance
    output_dir = tempfile.mkdtemp()
    turing_test = BioAcousticTuringTest(
        subject_id='demo_marmoset_001',
        species='marmoset',
        output_dir=output_dir
    )

    # Set phase
    turing_test.set_phase('concatenative_baseline')

    print("\n📢 Loading natural phee calls (concatenative)...")

    # Load natural stimuli (simulated for demo)
    sample_rates = [7800, 8200, 8500, 8800, 9100]  # Different F0 values
    for i, f0 in enumerate(sample_rates):
        audio = create_simulated_phee_call(22050, f0, 100.0)
        turing_test.add_stimulus(f'natural_phee_{f0}', audio.tolist(), 'concatenative')
        print(f"   Loaded: natural_phee_{f0} (F0={f0}Hz)")

    print("\n🔊 Running trials with natural stimuli...")

    # Run 10 trials
    response_count = 0
    for i in range(10):
        stimulus_id = f'natural_phee_{sample_rates[i % len(sample_rates)]}"
        result = turing_test.run_trial(stimulus_id)

        if result['has_response']:
            response_count += 1

        if (i + 1) % 5 == 0:
            print(f"   Completed {i + 1}/10 trials, {response_count} responses")

    # Get results
    results = turing_test.get_results()
    phase_results = results['concatenative_baseline']

    response_rate = np.mean(phase_results['responses'])
    print(f"\n📊 Concatenative Baseline Results:")
    print(f"   Total trials: {len(phase_results['trials'])}")
    print(f"   Response rate: {response_rate:.1%}")
    print(f"   Mean latency: {np.mean(phase_results['latencies_ms']):.1f}ms")

    return turing_test, response_rate


def demo_granular_synthesis(turing_test):
    """
    Demo Phase 2: Granular Synthesis

    Test response to granular-synthesized vocalizations at different
    pitch shifts. If granular synthesis preserves formant structure,
    animals should respond similarly to natural recordings.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: GRANULAR SYNTHESIS (Pitch-Shifted Variants)")
    print("=" * 80)

    if GranularConcatenativeSynthesizer is None:
        print("\n⚠️  Rust synthesizer not available, using Python fallback")

    # Set phase
    turing_test.set_phase('granular_synthesis')

    print("\n🎛️  Generating granular-synthesized stimuli...")

    # Create source audio (base phee call)
    source_audio = create_simulated_phee_call(22050, 8500, 100.0)

    # Generate pitch-shifted variants
    pitch_shifts = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

    if GranularConcatenativeSynthesizer:
        synthesizer = GranularConcatenativeSynthesizer(sample_rate=22050)

        for shift in pitch_shifts:
            # Load source
            synthesizer.load_source(source_audio.tolist())

            # Set pitch shift
            synthesizer.set_pitch_shift(shift)

            # Synthesize
            granular_audio_list = synthesizer.synthesize(100.0)
            granular_audio = np.array(granular_audio_list, dtype=np.float32)

            stimulus_id = f'granular_phee_shift_{shift:.2f}'
            turing_test.add_stimulus(stimulus_id, granular_audio.tolist(), 'granular')

            print(f"   Generated: {stimulus_id} (pitch shift: {shift:.2f}x)")
    else:
        # Python fallback: just pitch-shift the source
        for shift in pitch_shifts:
            # Simple resampling for pitch shift
            from scipy import signal
            num_samples = int(len(source_audio) / shift)
            granular_audio = signal.resample(source_audio, num_samples)

            stimulus_id = f'granular_phee_shift_{shift:.2f}'
            turing_test.add_stimulus(stimulus_id, granular_audio.tolist(), 'granular')

            print(f"   Generated: {stimulus_id} (pitch shift: {shift:.2f}x) [Python fallback]")

    print("\n🔊 Running trials with granular stimuli...")

    # Run trials
    stimulus_ids = [f'granular_phee_shift_{shift:.2f}' for shift in pitch_shifts]
    response_count = 0

    for i, stimulus_id in enumerate(stimulus_ids):
        result = turing_test.run_trial(stimulus_id)

        if result['has_response']:
            response_count += 1

        if (i + 1) % 3 == 0:
            print(f"   Completed {i + 1}/{len(stimulus_ids)} trials, {response_count} responses")

    # Get results
    results = turing_test.get_results()
    phase_results = results['granular_synthesis']

    response_rate = np.mean(phase_results['responses'])
    print(f"\n📊 Granular Synthesis Results:")
    print(f"   Total trials: {len(phase_results['trials'])}")
    print(f"   Response rate: {response_rate:.1%}")
    print(f"   Mean latency: {np.mean(phase_results['latencies_ms']):.1f}ms")

    return turing_test, response_rate


def demo_statistical_analysis(turing_test):
    """
    Demo Phase 3: Statistical Analysis

    Compare response rates between concatenative and granular
    conditions to evaluate the Turing test hypothesis.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    print("=" * 80)

    # Get hypothesis evaluation
    hypothesis = turing_test.evaluate_hypothesis()

    print("\n📈 Statistical Results:")
    print(f"   Null Hypothesis: {hypothesis['null_hypothesis']}")
    print(f"   Alternative: {hypothesis['alternative_hypothesis']}")
    print(f"   Conclusion: {hypothesis['conclusion']}")
    print(f"   Interpretation: {hypothesis['interpretation']}")

    print(f"\n   Response Rates:")
    print(f"   - Concatenative: {hypothesis['concatenative_response_rate']:.1%}")
    print(f"   - Granular: {hypothesis['granular_response_rate']:.1%}")

    if hypothesis.get('p_value'):
        print(f"\n   Statistical Test: {hypothesis.get('statistical_test', 'N/A')}")
        print(f"   P-value: {hypothesis['p_value']:.3f}")
        print(f"   Alpha: 0.05")
        print(f"   Significant at p<0.05: {hypothesis['p_value'] < 0.05}")

    print("\n" + "=" * 80)
    if hypothesis.get('passed'):
        print("✅ TURING TEST PASSED!")
        print("=" * 80)
        print("\n🎉 Animals CANNOT DISTINGUISH between natural and granular!")
        print("\n   This confirms that granular synthesis achieves bio-acoustic")
        print("   fidelity sufficient for animal behavior experiments.")
        print("\n💡 Implications:")
        print("   - Granular synthesis can be used for systematic parameter variation")
        print("   - Enables controlled experiments impossible with natural recordings")
        print("   - Preserves formant structure while allowing pitch/time manipulation")
    else:
        print("❌ TURING TEST FAILED")
        print("=" * 80)
        print("\n⚠️  Animals CAN DISTINGUISH between natural and granular")
        print("\n   Possible reasons:")
        print("   1. Granular synthesis parameters need tuning")
        print("   2. Grain size or window function not optimal")
        print("   3. Pitch shift artifacts too pronounced")
        print("   4. Need formant-preserving pitch shifting algorithm")

    print("=" * 80)

    return hypothesis


def main():
    """Run complete Bio-Acoustic Turing Test demo."""
    print("=" * 80)
    print("BIO-ACOUSTIC TURING TEST DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo simulates a complete Turing test to validate that")
    print("granular synthesis achieves bio-acoustic fidelity.")
    print("\nHypothesis: Animals cannot distinguish between natural")
    print("            and granular-synthesized vocalizations")

    # Phase 1: Concatenative baseline
    turing_test, concat_rate = demo_concatenative_baseline()

    # Phase 2: Granular synthesis
    turing_test, granular_rate = demo_granular_synthesis(turing_test)

    # Phase 3: Statistical analysis
    hypothesis = demo_statistical_analysis(turing_test)

    # Save results
    output_file = Path(turing_test.output_dir) / 'turing_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'subject_id': turing_test.subject_id,
            'species': turing_test.species,
            'concatenative_response_rate': concat_rate,
            'granular_response_rate': granular_rate,
            'hypothesis_result': hypothesis,
            'timestamp': str(np.datetime64('now'))
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    print("\n🎯 NEXT STEPS:")
    if hypothesis.get('passed'):
        print("   1. ✅ Granular synthesis validated with simulated data")
        print("   2. Run live animal experiment with real subjects")
        print("   3. Publish results demonstrating bio-acoustic validity")
    else:
        print("   1. ⚠️  Review granular synthesis parameters")
        print("   2. Consider smaller pitch shifts")
        print("   3. Test with grain sizes 10-30ms")
        print("   4. Re-run validation when parameters are optimized")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
