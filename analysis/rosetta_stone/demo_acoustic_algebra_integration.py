"""
Acoustic Algebra Integration Demo
================================

Demonstrates how High-Dimensional Acoustic Algebra integrates into the
phrase discovery and synthesis workflow.

**Pipeline Integration:**

┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Audio + Annotations                                   │
│  (Input: WAV files + ELAN/Praat Labels)                         │
└────────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Phrase Discovery + Contextual Map                      │
│  (DBSCAN Clustering + Annotation Association)                  │
│                                                              │
│  🆕 ALGEBRA ROLE 1: DEFINING SEMANTIC VECTORS                │
│  • Calculate "Context Centroids"                                  │
│    Vector_Aggression = Mean(17D_vectors for all "Agg" phrases)   │
│  • Calculate "Context Variance"                                    │
│    How spread out is "Aggression?"                                 │
└────────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  CONTEXTUAL VECTOR MAP      │
         │ (Aggression = +0.5 Jitter,   │
         │  -10ms Duration, etc.)       │
         └───────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Context-Aware Synthesis                                │
│  (Granular Concatenative Engine)                                │
│                                                              │
│  🆕 ALGEBRA ROLE 2: GRADIENT GENERATION                      │
│  • Input: Intent="Aggression", Intensity=0.7                      │
│  • Math: V_target = V_neutral + (V_agg - V_neutral) * 0.7         │
│  • Output: "Virtual Phrase" (70% Aggressive)                     │
└─────────────────────────────────────────────────────────────────┘

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.rosetta_stone.contextual_map import ContextualMap
from analysis.rosetta_stone.high_dimensional_acoustic_algebra import (
    AcousticFeatureVector17,
)


def create_mock_annotated_phrases():
    """
    Create mock phrases with annotation labels for demonstration.

    In production, this would come from:
    - ELAN annotation files
    - Praat TextGrid files
    - JSON annotations from annotation_loader.py
    """
    phrases = {}

    # Contact phrases (baseline)
    phrases['contact_001'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6500, duration_ms=70, attack_ms=0.010, decay_ms=0.050,
            f0_range_hz=400, vibrato_rate_hz=8.0, vibrato_depth_hz=50.0,
            jitter=0.02, shimmer=0.03, harmonicity_hnr=20.0, spectral_flatness=0.1,
            spectral_centroid_hz=7000.0, spectral_rolloff_hz=13000.0,
            bandwidth_hz=5000.0, slope_db_per_octave=-8.0,
            rms_db=-20.0, peak_amplitude=0.15
        ),
        'context': 'contact',
        'audio_file': 'contact_001.wav'
    }

    phrases['contact_002'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6450, duration_ms=75, attack_ms=0.012, decay_ms=0.045,
            f0_range_hz=450, vibrato_rate_hz=7.5, vibrato_depth_hz=45.0,
            jitter=0.018, shimmer=0.028, harmonicity_hnr=22.0, spectral_flatness=0.12,
            spectral_centroid_hz=7200.0, spectral_rolloff_hz=12500.0,
            bandwidth_hz=5200.0, slope_db_per_octave=-7.5,
            rms_db=-19.0, peak_amplitude=0.14
        ),
        'context': 'contact',
        'audio_file': 'contact_002.wav'
    }

    phrases['contact_003'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6550, duration_ms=68, attack_ms=0.008, decay_ms=0.055,
            f0_range_hz=380, vibrato_rate_hz=8.5, vibrato_depth_hz=48.0,
            jitter=0.022, shimmer=0.032, harmonicity_hnr=18.0, spectral_flatness=0.11,
            spectral_centroid_hz=6800.0, spectral_rolloff_hz=12800.0,
            bandwidth_hz=4800.0, slope_db_per_octave=-8.5,
            rms_db=-21.0, peak_amplitude=0.13
        ),
        'context': 'contact',
        'audio_file': 'contact_003.wav'
    }

    # Aggression phrases (target)
    phrases['aggression_001'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6100, duration_ms=55, attack_ms=0.005, decay_ms=0.030,
            f0_range_hz=3500, vibrato_rate_hz=12.0, vibrato_depth_hz=150.0,
            jitter=0.08, shimmer=0.05, harmonicity_hnr=5.0, spectral_flatness=0.3,
            spectral_centroid_hz=8000.0, spectral_rolloff_hz=15000.0,
            bandwidth_hz=8000.0, slope_db_per_octave=-4.0,
            rms_db=-15.0, peak_amplitude=0.25
        ),
        'context': 'aggression',
        'audio_file': 'aggression_001.wav'
    }

    phrases['aggression_002'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6000, duration_ms=50, attack_ms=0.004, decay_ms=0.025,
            f0_range_hz=3800, vibrato_rate_hz=11.0, vibrato_depth_hz=140.0,
            jitter=0.075, shimmer=0.045, harmonicity_hnr=6.0, spectral_flatness=0.28,
            spectral_centroid_hz=8200.0, spectral_rolloff_hz=14500.0,
            bandwidth_hz=7500.0, slope_db_per_octave=-4.5,
            rms_db=-14.0, peak_amplitude=0.23
        ),
        'context': 'aggression',
        'audio_file': 'aggression_002.wav'
    }

    phrases['aggression_003'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6050, duration_ms=58, attack_ms=0.006, decay_ms=0.035,
            f0_range_hz=3200, vibrato_rate_hz=11.5, vibrato_depth_hz=145.0,
            jitter=0.078, shimmer=0.048, harmonicity_hnr=5.5, spectral_flatness=0.29,
            spectral_centroid_hz=8100.0, spectral_rolloff_hz=14800.0,
            bandwidth_hz=7800.0, slope_db_per_octave=-4.2,
            rms_db=-14.5, peak_amplitude=0.24
        ),
        'context': 'aggression',
        'audio_file': 'aggression_003.wav'
    }

    # Food phrases
    phrases['food_001'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6300, duration_ms=65, attack_ms=0.008, decay_ms=0.040,
            f0_range_hz=600, vibrato_rate_hz=9.0, vibrato_depth_hz=60.0,
            jitter=0.025, shimmer=0.035, harmonicity_hnr=15.0, spectral_flatness=0.15,
            spectral_centroid_hz=7500.0, spectral_rolloff_hz=13500.0,
            bandwidth_hz=6000.0, slope_db_per_octave=-6.0,
            rms_db=-18.0, peak_amplitude=0.18
        ),
        'context': 'food',
        'audio_file': 'food_001.wav'
    }

    phrases['food_002'] = {
        'vector': AcousticFeatureVector17(
            mean_f0_hz=6350, duration_ms=62, attack_ms=0.009, decay_ms=0.038,
            f0_range_hz=580, vibrato_rate_hz=8.5, vibrato_depth_hz=55.0,
            jitter=0.023, shimmer=0.033, harmonicity_hnr=16.0, spectral_flatness=0.14,
            spectral_centroid_hz=7400.0, spectral_rolloff_hz=13300.0,
            bandwidth_hz=5800.0, slope_db_per_octave=-6.5,
            rms_db=-17.5, peak_amplitude=0.17
        ),
        'context': 'food',
        'audio_file': 'food_002.wav'
    }

    return phrases


def demo_discovery_phase():
    """
    **STEP 2**: Phrase Discovery + Contextual Map

    Calculate semantic centroids from annotated phrases.
    """
    print("\n" + "="*80)
    print("STEP 2: DISCOVERY PHASE - Calculate Contextual Centroids")
    print("="*80)

    # Load annotated phrases (mock data for demo)
    phrases = create_mock_annotated_phrases()

    # Extract vectors and labels
    phrase_vectors = {k: v['vector'] for k, v in phrases.items()}
    context_labels = {k: v['context'] for k, v in phrases.items()}

    print(f"\n📊 Loaded {len(phrases)} annotated phrases:")
    for phrase_id, phrase_data in phrases.items():
        print(f"  • {phrase_id}: {phrase_data['context'].upper()}")

    # Create contextual map and calculate centroids
    map_obj = ContextualMap()
    centroids = map_obj.calculate_context_centroids(phrase_vectors, context_labels)

    print(f"\n✅ Calculated {len(centroids)} contextual centroids")

    # Show what each centroid "means"
    print("\n" + "-"*80)
    print("CONTEXTUAL SEMANTIC ANALYSIS")
    print("-"*80)

    for ctx_name in ['contact', 'aggression', 'food']:
        if ctx_name in centroids:
            centroid = centroids[ctx_name]
            print(f"\n{ctx_name.upper()} CONTEXT:")
            print(f"  Sample Count: {centroid.sample_count}")
            print(f"  Centroid Vector: {centroid.centroid_vector}")

            # Calculate context delta from baseline
            if ctx_name != map_obj.baseline_context:
                delta = map_obj.calculate_context_delta(ctx_name, map_obj.baseline_context)
                print(f"  Delta from Baseline: {delta}")

                # Show top 3 differentiating features
                delta_vec = delta.to_numpy()
                feature_names = delta.feature_names()
                top_delta_idx = np.argsort(np.abs(delta_vec))[-3:][::-1]

                print("  Top 3 Differentiating Features:")
                for idx in top_delta_idx:
                    print(f"    - {feature_names[idx]}: {delta_vec[idx]:+.2f}")

    return map_obj, phrase_vectors


def demo_synthesis_phase(map_obj, phrase_vectors):
    """
    **STEP 3**: Context-Aware Synthesis with Gradient Generation

    Generate "Virtual Phrases" at specified intensities.
    """
    print("\n" + "="*80)
    print("STEP 3: SYNTHESIS PHASE - Gradient Generation")
    print("="*80)

    print("\n🎯 Standard Retrieval (Without Algebra):")
    print("  Request: 'Aggressive call'")
    print("  Action: Pick random phrase from 'aggression' bucket")
    print("  Result: You get FULL aggression (cannot get 30% aggression)")

    print("\n🎯 Algebra-Enhanced Synthesis (With Gradient Generation):")
    print("  Request: 'Aggressive call at 30% intensity'")
    print("  Action: Interpolate between Contact and Aggression vectors")
    print("  Result: You get a nuanced '30% Aggressive' virtual phrase")

    # Generate graded phrases
    print("\n" + "-"*80)
    print("GENERATING GRADED PHRASES (Contact → Aggression)")
    print("-"*80)

    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

    for intensity in intensities:
        # Generate virtual phrase
        virtual = map_obj.generate_graded_phrase('aggression', intensity)

        # Find nearest real phrase
        nearest_key, nearest_vec, distance = map_obj.find_nearest_real_phrase(
            virtual, phrase_vectors
        )

        # Calculate feature differences
        virtual_vec = virtual.to_numpy()
        nearest_vec_numpy = nearest_vec.to_numpy()
        diff = virtual_vec - nearest_vec_numpy

        # Show key features that differ
        feature_names = virtual.feature_names()
        key_features = ['mean_f0_hz', 'duration_ms', 'attack_ms', 'jitter', 'harmonicity_hnr']

        print(f"\n🎯 Intensity {intensity*100:.0f}%:")
        print(f"  Virtual:  F0={virtual.mean_f0_hz:.0f}Hz, "
              f"Dur={virtual.duration_ms:.1f}ms, "
              f"Attack={virtual.attack_ms*1000:.1f}ms, "
              f"Jitter={virtual.jitter:.3f}")
        print(f"  Nearest: {nearest_key} (distance: {distance:.3f})")

        # Show if we need to apply micro-dynamic extrapolation
        if distance > 1.0:
            print("  ⚠️  Large distance - apply micro-dynamic extrapolation:")
            for feat in key_features:
                idx = feature_names.index(feat)
                if abs(diff[idx]) > 0.01:
                    print(f"     • {feat}: {diff[idx]:+.3f}")


def demo_threshold_test():
    """
    **SCIENTIFIC BENEFIT**: The "Threshold Test"

    Test hypothesis: Animals perceive emotion as a continuous continuum,
    not discrete states.
    """
    print("\n" + "="*80)
    print("SCIENTIFIC APPLICATION: The Threshold Test")
    print("="*80)

    print("""
🧪 Hypothesis: Animals perceive emotion as a continuous continuum

📊 Experiment Design:

  Condition A (Baseline):  Intensity 0.0  → Contact
  Condition B (Midpoint):    Intensity 0.5  → Mild Aggression
  Condition C (Full):       Intensity 1.0  → Full Aggression

📈 Measurement: Plot behavioral response vs. Intensity %

  If Linear:  Animal perceives a GRADIENT (Proof of Acoustic Continuum)
  If Step:    Animal perceives a CATEGORY (Proof of Discrete Semantics)

💡 This experiment was IMPOSSIBLE before because:
    - Old system: Only 3 discrete levels (contact, aggression, food)
    - New system: Infinite precision via acoustic algebra
    """)

    print("\n🎯 Example Experimental Stimuli:")

    map_obj, phrase_vectors = demo_discovery_phase()

    print("\n📊 Experimental Stimuli Generated:")

    for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
        virtual = map_obj.generate_graded_phrase('aggression', intensity)

        # Calculate acoustic distance from baseline
        baseline = map_obj.centroids[map_obj.baseline_context].centroid_vector
        delta = map_obj.algebra.subtract(virtual, baseline)

        print(f"\n  Intensity {intensity*100:.0f}%:")
        print(f"    F0: {virtual.mean_f0_hz:.0f} Hz (Δ from baseline: {delta.mean_f0_hz:+.0f})")
        print(f"    Duration: {virtual.duration_ms:.1f} ms (Δ: {delta.duration_ms:+.1f})")
        print(f"    Attack: {virtual.attack_ms*1000:.1f} ms (Δ: {delta.attack_ms*1000:+.1f})")
        print(f"    Jitter: {virtual.jitter:.3f} (Δ: {delta.jitter:+.3f})")

    print("\n" + "="*80)


def main():
    """Run complete demonstration."""
    print("\n" + "="*80)
    print("ACOUSTIC ALGEBRA INTEGRATION DEMO")
    print("="*80)

    print("""
🎯 Acoustic Algebra transforms the pipeline from:
  • Discrete Retrieval → Binary: Aggressive vs. Not Aggressive
  • Continuous Generation → Gradient: 50% Aggressive

📍 Integration Points:
  1. Discovery Phase: Defines "Semantic Vectors" of contexts
  2. Synthesis Phase: Generates "Virtual Phrases" at nuanced intensities
    """)

    # Run demos
    map_obj, phrase_vectors = demo_discovery_phase()
    demo_synthesis_phase(map_obj, phrase_vectors)
    demo_threshold_test()

    print("\n" + "="*80)
    print("✅ INTEGRATION COMPLETE")
    print("="*80)

    print("""
📦 Files Created:
  • analysis/rosetta_stone/high_dimensional_acoustic_algebra.py
  • analysis/rosetta_stone/contextual_map.py

🔗 Integration Points:
  • Import ContextualMap in audio_aware_grammar_discovery.py
  • Call calculate_context_centroids() during discovery phase
  • Call generate_graded_phrase() during synthesis phase
  • Use find_nearest_real_phrase() to get source audio buffer

🧪 Ready for Threshold Test:
  • Generates continuous gradient from 0% to 100% intensity
  • Enables "continuum of meaning" experiments
  • Proves whether animals perceive gradients or categories
    """)


if __name__ == "__main__":
    main()
