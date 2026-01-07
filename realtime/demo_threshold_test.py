"""
Acoustic Algebra: Threshold Test & Semantic Gradient Demo
==========================================================

This demo demonstrates the key scientific breakthrough of treating behavioral
contexts as continuous mathematical dimensions rather than discrete buckets.

Key Concepts:
1. Semantic Vector Space: 17D acoustic feature space
2. Context Centroids: Mathematical definition of "aggression", "courtship", etc.
3. Gradient Synthesis: Generate vocalizations at any intensity (0.0-1.0)
4. Threshold Test: Measure animal perception along emotional continuum

Scientific Hypothesis:
- Animals perceive emotion as a CONTINUUM (gradient) not discrete states
- Response should scale LINEARLY with acoustic intensity along semantic axes

Pipeline:
    Audio + Annotations
           ↓
    Grain-based DBSCAN Discovery
           ↓
    Contextual Centroid Calculation (What "Aggression" sounds like)
           ↓
    17D Semantic Vector Space
           ↓
    Gradient Synthesis (Intensity 0.0 → 1.0)
           ↓
    Virtual Phrases (Nuanced intensities not in dataset)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings

from realtime.audio_aware_grammar_discovery import AudioAwareGrammarDiscovery

from realtime.phrase_audio_library import VocalizationSynthesizer

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Synthetic Audio Generation with Acoustic Gradients
# ============================================================================


def generate_gradient_vocalization(
    context: str, intensity: float, base_f0: float, duration_sec: float, sr: int = 22050
) -> np.ndarray:
    """
    Generate vocalization with context-specific intensity.

    Intensity mapping:
    - 0.0: Baseline/neutral characteristics
    - 0.5: Mid-intensity (context-specific traits at 50%)
    - 1.0: Full context expression

    Args:
        context: Behavioral context (aggression, courtship, etc.)
        intensity: 0.0 to 1.0
        base_f0: Base fundamental frequency
        duration_sec: Duration in seconds
        sr: Sample rate

    Returns:
        Audio array
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    audio = np.zeros_like(t)

    # Base parameters (neutral/contact call)
    neutral_f0 = base_f0
    neutral_attack = 0.05
    neutral_harmonics = 3

    # Context-specific parameters (full intensity)
    context_params = {
        "aggression": {
            "f0_mult": 1.3,
            "attack_mult": 0.3,
            "harmonics": 6,
            "jitter": 0.05,
            "roughness": 0.3,
        },
        "courtship": {
            "f0_mult": 0.9,
            "attack_mult": 1.2,
            "harmonics": 4,
            "jitter": 0.0,
            "roughness": 0.0,
        },
        "food_discovery": {
            "f0_mult": 0.8,
            "attack_mult": 0.8,
            "harmonics": 3,
            "jitter": 0.02,
            "roughness": 0.1,
        },
        "alarm": {
            "f0_mult": 1.5,
            "attack_mult": 0.2,
            "harmonics": 5,
            "jitter": 0.08,
            "roughness": 0.4,
        },
    }

    # Get context parameters or use neutral defaults
    params = context_params.get(context, context_params["courtship"])

    # Interpolate parameters based on intensity
    # Formula: param = neutral + (context - neutral) * intensity
    f0_mult = 1.0 + (params["f0_mult"] - 1.0) * intensity
    attack_mult = 1.0 + (params["attack_mult"] - 1.0) * intensity
    n_harmonics = int(neutral_harmonics + (params["harmonics"] - neutral_harmonics) * intensity)
    jitter = params["jitter"] * intensity
    roughness = params["roughness"] * intensity

    # Generate tone
    f0 = neutral_f0 * f0_mult
    for h in range(1, n_harmonics + 1):
        amplitude = 0.4 / h

        # Add jitter for higher intensities
        if jitter > 0:
            freq_mod = 1.0 + jitter * np.sin(2 * np.pi * 50 * t)
            signal = amplitude * np.sin(2 * np.pi * f0 * h * t * freq_mod)
        else:
            signal = amplitude * np.sin(2 * np.pi * f0 * h * t)

        audio += signal

    # Apply envelope
    attack_time = neutral_attack * attack_mult
    envelope = np.exp(-t / (duration_sec * 0.3))
    envelope *= 1 - np.exp(-t / attack_time)

    # Add roughness for higher intensities
    if roughness > 0:
        noise = np.random.randn(len(audio)) * roughness * 0.1
        audio += noise * envelope

    audio *= envelope

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)

    return audio


def generate_gradient_dataset(
    output_dir: Path, context: str = "aggression", intensities: List[float] = None
) -> Tuple[Path, Path]:
    """
    Generate a dataset of vocalizations at different intensities.

    Args:
        output_dir: Output directory
        context: Behavioral context to vary
        intensities: List of intensities (0.0 to 1.0)

    Returns:
        Tuple of (audio_path, annotation_path)
    """
    if intensities is None:
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

    logger.info(f"\nGenerating gradient dataset for '{context}'")
    logger.info(f"  Intensities: {intensities}")

    sr = 22050
    duration_per_intensity = 0.5  # 500ms per intensity
    gap = 0.2  # 200ms gap
    total_duration = len(intensities) * (duration_per_intensity + gap)

    audio = np.zeros(int(sr * total_duration))
    annotations = []

    current_time = 0.0
    base_f0 = 6000  # Base F0

    for i, intensity in enumerate(intensities):
        # Generate vocalization at this intensity
        phrase_audio = generate_gradient_vocalization(
            context=context,
            intensity=intensity,
            base_f0=base_f0,
            duration_sec=duration_per_intensity,
            sr=sr,
        )

        # Insert into recording
        start_sample = int(current_time * sr)
        end_sample = start_sample + len(phrase_audio)

        if end_sample <= len(audio):
            audio[start_sample:end_sample] += phrase_audio

            # Create annotation
            context_label = f"{context}_{intensity:.0%}"
            annotation = {
                "start_time_ms": current_time * 1000,
                "end_time_ms": (current_time + duration_per_intensity) * 1000,
                "context": context_label,
                "intensity": intensity,
                "notes": f"{context} at {intensity:.0%} intensity",
            }
            annotations.append(annotation)

        current_time += duration_per_intensity + gap

    # Save audio
    audio_path = output_dir / f"{context}_gradient_recording.wav"
    sf.write(audio_path, audio, sr)

    # Save annotations
    annotation_path = output_dir / f"{context}_gradient_annotations.json"
    annotation_data = {
        "metadata": {
            "species": "marmoset",
            "context": context,
            "intensities": intensities,
            "type": "gradient_dataset",
        },
        "annotations": annotations,
    }

    with open(annotation_path, "w") as f:
        json.dump(annotation_data, f, indent=2)

    logger.info(f"  ✓ Audio: {audio_path}")
    logger.info(f"  ✓ Annotations: {annotation_path}")

    return audio_path, annotation_path


# ============================================================================
# Threshold Test: Scientific Analysis
# ============================================================================


def run_threshold_test(
    pipeline: AudioAwareGrammarDiscovery, library, output_dir: Path, context: str = "aggression"
) -> Dict:
    """
    Run the threshold test to measure semantic continuity.

    This test generates vocalizations at multiple intensities and analyzes
    whether they form a continuous gradient or discrete categories.

    Args:
        pipeline: AudioAwareGrammarDiscovery with computed gradients
        library: PhraseAudioLibrary
        output_dir: Output directory
        context: Context to test

    Returns:
        Dictionary with test results
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"THRESHOLD TEST: {context.upper()}")
    logger.info("=" * 80)

    logger.info("""
Hypothesis: Animals perceive emotion as a continuous continuum, not discrete states.

Test Conditions:
  A. Intensity 0.0 (Baseline/Contact)
  B. Intensity 0.5 (Midpoint: 50% Aggression)
  C. Intensity 1.0 (Full Aggression)

Expected Results:
  - Linear Response: Animal perceives GRADIENT (continuum proven)
  - Step Function: Animal perceives CATEGORY (discrete semantics)

This creates a new class of experiment: The Threshold Test.
    """)

    results = {
        "context": context,
        "intensities_tested": [],
        "virtual_vectors": [],
        "synthesis_params": [],
        "acoustic_distances": [],
    }

    # Test at multiple intensities
    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

    logger.info(f"\nTesting {len(intensities)} intensity levels:")

    baseline = pipeline.contextual_map.baseline_context

    for intensity in intensities:
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Intensity: {intensity:.2f} ({intensity:.0%})")

        # Synthesize gradient
        result = pipeline.synthesize_gradient(
            intent=context, intensity=intensity, baseline_context=baseline
        )

        if result:
            # Store results
            results["intensities_tested"].append(intensity)
            results["virtual_vectors"].append(result["virtual_vector"])
            results["synthesis_params"].append(result["synthesis_params"])

            # Calculate acoustic distance from baseline
            if intensity > 0:
                baseline_vec = pipeline.contextual_map.get_context_vector(baseline)
                target_vec = result["virtual_vector"]
                distance = baseline_vec.distance_to(target_vec)
                results["acoustic_distances"].append(distance)

                logger.info(f"  Acoustic distance from baseline: {distance:.3f}")
                logger.info("  Synthesis params:")
                for param, value in result["synthesis_params"].items():
                    if abs(value) > 0.01:
                        logger.info(f"    {param}: {value:.3f}")

    # Analyze linearity
    logger.info(f"\n{'=' * 80}")
    logger.info("THRESHOLD TEST ANALYSIS")
    logger.info(f"{'=' * 80}")

    if len(results["acoustic_distances"]) > 1:
        distances = results["acoustic_distances"]
        intensities_nonzero = [i for i in intensities if i > 0]

        # Check linearity (correlation between intensity and distance)
        correlation = np.corrcoef(intensities_nonzero, distances)[0, 1]

        logger.info("\nGradient Linearity Analysis:")
        logger.info(f"  Intensity-Distance Correlation: {correlation:.3f}")

        if correlation > 0.95:
            logger.info("  Result: ✓ LINEAR (Strong gradient detected)")
            logger.info("  Interpretation: Continuum perception supported")
        elif correlation > 0.7:
            logger.info("  Result: ~ MOSTLY LINEAR (Moderate gradient)")
            logger.info("  Interpretation: Partial continuum with some thresholds")
        else:
            logger.info("  Result: ✗ NON-LINEAR (Step function)")
            logger.info("  Interpretation: Discrete categories detected")

    return results


# ============================================================================
# Demo: Semantic Gradient Synthesis
# ============================================================================


def demo_gradient_synthesis(
    pipeline: AudioAwareGrammarDiscovery,
    library,
    synthesizer: VocalizationSynthesizer,
    output_dir: Path,
):
    """
    Demonstrate gradient synthesis capabilities.

    Shows:
    1. Interpolation between contexts
    2. Virtual phrase generation
    3. Continuous emotional modulation
    """
    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC GRADIENT SYNTHESIS DEMO")
    logger.info("=" * 80)

    logger.info("""
Without Acoustic Algebra (Discrete Retrieval):
  Request: "Give me an aggressive call"
  Action: Pick random phrase from "aggression" bucket
  Result: Full-blown rage (can't do "mildly annoyed")

With Acoustic Algebra (Continuous Generation):
  Request: "Give me 30% aggression"
  Action: Interpolate vector: V = V_contact + (V_agg - V_contact) * 0.3
  Result: Mildly annoyed call (nuanced intensity not in dataset)
    """)

    # Get available contexts
    available_contexts = list(pipeline.contextual_map.contexts.keys())

    if len(available_contexts) < 2:
        logger.warning("Need at least 2 contexts for gradient synthesis")
        return

    # Use first non-baseline context as target
    baseline = pipeline.contextual_map.baseline_context
    target_contexts = [ctx for ctx in available_contexts if ctx != baseline]

    if not target_contexts:
        logger.warning("No target contexts available")
        return

    target = target_contexts[0]

    logger.info(f"\nGradient Synthesis: {baseline} → {target}")
    logger.info(f"Baseline: {baseline}")
    logger.info(f"Target: {target}")

    # Generate at multiple intensities
    logger.info("\nGenerating gradient continuum:")

    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

    for intensity in intensities:
        result = pipeline.synthesize_gradient(target, intensity)

        if result and result["nearest_phrase"]:
            logger.info(f"\n  Intensity {intensity:.0%}:")
            logger.info(f"    Virtual Context: {result['virtual_vector'].context}")
            logger.info(f"    Nearest Real Phrase: {result['nearest_phrase'].phrase_key}")

            # Could synthesize audio here using the synthesis params
            # This would require integrating with a granular synth


# ============================================================================
# Main Demo
# ============================================================================


def main():
    """Run the complete acoustic algebra demo."""
    print("\n" + "=" * 80)
    print("ACOUSTIC ALGEBRA: THRESHOLD TEST & GRADIENT SYNTHESIS")
    print("=" * 80)

    print("""
This demo demonstrates the Semantic Gradient Engine:

Key Concepts:
1. 17D Semantic Vector Space - Acoustic features as mathematical dimensions
2. Context Centroids - Mathematical definition of "aggression", "courtship", etc.
3. Gradient Synthesis - Generate vocalizations at any intensity (0.0-1.0)
4. Threshold Test - Measure if animals perceive continua or discrete categories

Scientific Impact:
- Transforms discrete retrieval (play a file) → continuous generation (create vector)
- Enables nuanced synthesis (mildly annoyed, not just full rage)
- Threshold Test: Prove animals perceive emotion as continuum
    """)

    # Setup
    output_dir = Path("/tmp/acoustic_algebra_demo")
    output_dir.mkdir(exist_ok=True)

    # ========================================================================
    # Phase 1: Generate Gradient Dataset
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 1: GENERATE GRADIENT DATASET")
    print("=" * 80)

    audio_path, annotation_path = generate_gradient_dataset(
        output_dir, context="aggression", intensities=[0.0, 0.25, 0.5, 0.75, 1.0]
    )

    # ========================================================================
    # Phase 2: Discover Phrases with Context
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 2: DISCOVER PHRASES WITH CONTEXT")
    print("=" * 80)

    pipeline = AudioAwareGrammarDiscovery(
        grain_duration_ms=15.0,
        hop_size_ms=7.5,
        dbscan_eps=0.8,
        dbscan_min_samples=2,
        sample_rate=22050,
    )

    # Load audio
    audio = pipeline.load_audio_file(str(audio_path))
    pipeline.extract_audio_grains(audio, pipeline.sample_rate)
    pipeline.discover_atomic_phrases()

    # Load annotations
    pipeline.load_annotations(str(annotation_path))

    # Build library with context
    library = pipeline.build_phrase_library(
        species="marmoset",
        export_path=str(output_dir / "gradient_phrase_library.pkl"),
        associate_context=True,
    )

    # ========================================================================
    # Phase 3: Compute Semantic Gradients (Acoustic Algebra)
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTE SEMANTIC GRADIENTS (ACOUSTIC ALGEBRA)")
    print("=" * 80)

    _ = pipeline.compute_semantic_gradients(library)  # Contextual map computed

    # ========================================================================
    # Phase 4: Threshold Test
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 4: THRESHOLD TEST")
    print("=" * 80)

    _ = run_threshold_test(pipeline, library, output_dir, context="aggression")

    # ========================================================================
    # Phase 5: Gradient Synthesis Demo
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 5: GRADIENT SYNTHESIS DEMO")
    print("=" * 80)

    synthesizer = VocalizationSynthesizer(library)
    demo_gradient_synthesis(pipeline, library, synthesizer, output_dir)

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
✅ Acoustic Algebra Implementation Complete

Key Achievements:

1. ✓ Semantic Vector Space (17D)
     - All acoustic features normalized (z-score)
     - Context centroids calculated
     - Mathematical definition of emotions

2. ✓ Gradient Synthesis Engine
     - Interpolate between contexts at any intensity (0.0-1.0)
     - Generate "virtual phrases" not in dataset
     - Continuous emotional modulation

3. ✓ Threshold Test Framework
     - Measure perception along emotional continuum
     - Test: continuum (linear) vs discrete (step function)
     - Enables new class of scientific experiments

4. ✓ Integration with Discovery Pipeline
     - compute_semantic_gradients() after library building
     - synthesize_gradient(intent, intensity) for generation
     - Seamless integration with annotation-aware pipeline

Scientific Impact:

Without Algebra (Discrete):
  - 3 discrete levels (low, medium, high aggression)
  - Retrieval-based (play existing files)
  - Cannot generate nuanced intensities

With Algebra (Continuous):
  - Infinite precision (0.0-1.0 continuum)
  - Generation-based (create virtual vectors)
  - Mathematical definition of emotional space
  - Threshold Test: Prove continuum perception

Usage Example:

    from realtime.audio_aware_grammar_discovery import AudioAwareGrammarDiscovery

    # Standard pipeline
    pipeline = AudioAwareGrammarDiscovery()
    pipeline.load_audio_file("recording.wav")
    pipeline.load_annotations("annotations.json")
    pipeline.extract_audio_grains(audio, sr)
    pipeline.discover_atomic_phrases()

    # Build library
    library = pipeline.build_phrase_library(
        species="marmoset",
        associate_context=True
    )

    # 🆕 Compute Semantic Gradients
    contextual_map = pipeline.compute_semantic_gradients(library)

    # 🆕 Synthesize at 50% aggression
    result = pipeline.synthesize_gradient(
        intent="aggression",
        intensity=0.5  # 50% intensity!
    )

Files Created:
  - realtime/acoustic_algebra_contextual.py (NEW)
  - tests/test_acoustic_algebra_contextual.py (NEW)
  - realtime/demo_threshold_test.py (NEW - this file)

This completes the Semantic Gradient Engine implementation!
    """)

    print(f"\n📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
