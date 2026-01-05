"""
Demo Script for Semiotic Detection Engine

This script demonstrates how to use the semiotic analysis system to understand
the cognitive dimensions of animal communication.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_models import Species, Phrase, PhraseContext, VocalizationModality
from semiotics.semiotic_engine import (
    SemioticEngine, SemioticState, SemioticRelation,
    SemioticContext, AcousticFeatures
)
import numpy as np


def demo_deceptive_semiotics():
    """Demonstrate deceptive semiotics detection"""
    print("="*60)
    print("DECEPTIVE SEMIOTICS DEMONSTRATION")
    print("="*60)

    engine = SemioticEngine()

    # Create a fake alarm call (potential deception)
    fake_alarm = Phrase(
        phrase_key="FAKE_PREDATOR_ALARM",
        signature="fake_threat_signal",
        species=Species.MARMOSET,
        modality=VocalizationModality.HARMONIC,
        acoustic_features=AcousticFeatures(
            mean_f0_hz=12000.0,
            std_f0_hz=150.0,  # High variability
            min_f0_hz=11500.0,
            max_f0_hz=12500.0,
            f0_range_hz=1000.0,
            duration_frames=120,
            voiced_ratio=0.7,
            f0_slope=0.0,
            modulation_rate=0.0,
            acoustic_variance=0.8,
            mean_duration_ms=120.0
        ),
        total_occurrences=3,  # Rare usage
        contexts=[
            PhraseContext("predator", 2, 66.7),
            PhraseContext("aggression", 1, 33.3)
        ],
        social_contexts={
            "dominance": True,
            "resource_competition": True
        }
    )

    # Create context suggesting deception
    context = SemioticContext(
        species=Species.MARMOSET,
        acoustic_features=fake_alarm.acoustic_features,
        social_context={
            "individual_id": "alpha_male",
            "dominance_rank": 1,
            "resource_control": "food_patch",
            "competitor_present": True,
            "no_immediate_threat": True  # Key deception indicator
        },
        behavioral_context={
            "current_behavior": "peaceful_foraging",
            "posture": "relaxed",
            "attention_direction": "away_from_competitor"
        }
    )

    # Analyze for deception
    result = engine.analyze_semiotics(fake_alarm, context)

    print(f"Phrase: {result.phrase_key}")
    print(f"Semiotic State: {result.semiotic_state.value}")
    print(f"Relation Type: {result.relation_type.value}")
    print(f"Deception Score: {result.deception_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Context Alignment: {result.context_alignment:.3f}")
    print(f"Behavioral Correlates: {result.behavioral_correlates}")

    if result.behavioral_correlates.get("context_misalignment"):
        print("⚠️  Context misalignment detected - potential deception!")

    print()


def demo_emergent_semiotics():
    """Demonstrate emergent semiotics detection"""
    print("="*60)
    print("EMERGENT SEMIOTICS DEMONSTRATION")
    print("="*60)

    engine = SemioticEngine()

    # Create an innovative call (potential emergence)
    innovative_call = Phrase(
        phrase_key="NOVEL_WHISTLE_PATTERN",
        signature="innovative_signal",
        species=Species.DOLPHIN,
        modality=VocalizationModality.WHISTLE,
        acoustic_features=AcousticFeatures(
            mean_f0_hz=10000.0,
            std_f0_hz=300.0,  # High variability suggests innovation
            min_f0_hz=9200.0,
            max_f0_hz=10800.0,
            f0_range_hz=1600.0,
            duration_frames=250,
            voiced_ratio=0.95,
            f0_slope=0.3,
            modulation_rate=25.0,
            acoustic_variance=0.6,
            mean_duration_ms=250.0
        ),
        total_occurrences=1,  # First occurrence
        contexts=[
            PhraseContext("novel_situation", 1, 100.0)
        ],
        is_compositional=True,
        phrase_components=["F0_8000_COMPONENT", "F0_12000_COMPONENT"]
    )

    # Create context suggesting emergence
    context = SemioticContext(
        species=Species.DOLPHIN,
        acoustic_features=innovative_call.acoustic_features,
        social_context={
            "novel_situation": "human_introduction",
            "first_observation": True,
            "innovation_context": "problem_solving",
            "social_learning": True,
            "observation_potential": 0.9
        },
        behavioral_context={
            "current_behavior": "exploration",
            "problem_solving": True,
            "social_experimentation": True
        }
    )

    # Analyze for emergence
    result = engine.analyze_semiotics(innovative_call, context)

    print(f"Phrase: {result.phrase_key}")
    print(f"Semiotic State: {result.semiotic_state.value}")
    print(f"Relation Type: {result.relation_type.value}")
    print(f"Emergence Score: {result.emergence_score:.3f}")
    print(f"Innovation Potential: {result.innovation_potential:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Cross-Modal Attention: {result.cross_modal_attention}")

    if result.semiotic_state == SemioticState.EMERGENT:
        print("✨ Emergent semiotics detected - innovation observed!")
        print(f"Interpretant Chain: {' → '.join(result.interpretant_chain)}")

    print()


def demo_directed_communication():
    """Demonstrate directed communication detection"""
    print("="*60)
    print("DIRECTED COMMUNICATION DEMONSTRATION")
    print("="*60)

    engine = SemioticEngine()

    # Create a targeted call
    targeted_call = Phrase(
        phrase_key="SPECIFIC_individual_whistle",
        signature="targeted_signal",
        species=Species.DOLPHIN,
        modality=VocalizationModality.WHISTLE,
        acoustic_features=AcousticFeatures(
            mean_f0_hz=12000.0,
            std_f0_hz=50.0,  # Consistent pattern
            min_f0_hz=11800.0,
            max_f0_hz=12200.0,
            f0_range_hz=400.0,
            duration_frames=180,
            voiced_ratio=0.98,
            f0_slope=0.0,
            modulation_rate=5.0,
            acoustic_variance=0.05,
            mean_duration_ms=180.0
        ),
        total_occurrences=45,
        contexts=[
            PhraseContext("specific_individual", 45, 100.0)
        ],
        social_contexts={
            "target_id": "dolphin_alpha",
            "social_bond": "strong",
            "communication_type": "directed"
        }
    )

    # Create context for directed communication
    context = SemioticContext(
        species=Species.DOLPHIN,
        acoustic_features=targeted_call.acoustic_features,
        social_context={
            "target_id": "dolphin_alpha",
            "social_relationship": "pod_member",
            "communication_goal": "coordination"
        },
        behavioral_context={
            "current_behavior": "cooperative_foraging",
            "attention_target": "dolphin_alpha",
            "joint_attention": True,
            "bilateral_coordination": True
        },
        communication_target="dolphin_alpha"
    )

    # Analyze for directed communication
    result = engine.analyze_semiotics(targeted_call, context)

    print(f"Phrase: {result.phrase_key}")
    print(f"Semiotic State: {result.semiotic_state.value}")
    print(f"Relation Type: {result.relation_type.value}")
    print(f"Directed Score: {result.directed_score:.3f}")
    print(f"Communication Target: {result.communication_target}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Behavioral Correlates: {result.behavioral_correlates}")

    if result.relation_type == SemioticRelation.DIRECTED:
        print(f"🎯 Directed communication to {result.communication_target} detected!")

    print()


def demo_cross_species_analysis():
    """Demonstrate cross-species semiotic patterns"""
    print("="*60)
    print("CROSS-SPECIES SEMIOTIC PATTERNS")
    print("="*60)

    engine = SemioticEngine()

    print("Semiotic patterns by species:")
    for species, patterns in engine.semiotic_patterns.items():
        if patterns:
            print(f"\n{species.value}:")
            for pattern in patterns[:3]:  # Show first 3 patterns
                print(f"  - {pattern.pattern_id}: {pattern.relation_type.value}")
                print(f"    Contexts: {', '.join(pattern.common_contexts[:2])}")
                print(f"    Frequency: {pattern.frequency}")

    print("\nCross-species comparison:")
    for species in [Species.MARMOSET, Species.DOLPHIN]:
        print(f"\n{species.value} communication patterns:")
        patterns = engine.semiotic_patterns.get(species, [])
        if patterns:
            relation_types = [p.relation_type.value for p in patterns]
            print(f"  Relation types: {set(relation_types)}")
            print(f"  Total patterns: {len(patterns)}")


def main():
    """Run all demo functions"""
    print("🧠 Semiotic Detection Engine Demo")
    print("================================")
    print("Transforming vocalization data into cognitive intelligence...")
    print()

    demo_deceptive_semiotics()
    demo_emergent_semiotics()
    demo_directed_communication()
    demo_cross_species_analysis()

    print("\n" + "="*60)
    print("SEMIOTIC ENGINE ANALYSIS COMPLETE")
    print("="*60)
    print("\nThe system now understands:")
    print("• Deceptive communication tactics")
    print("• Emergent innovative behaviors")
    print("• Directed targeted communication")
    print("• Cross-species semiotic patterns")
    print("\nYour vocalization system has evolved into a true Cognitive Intelligence Engine!")


if __name__ == "__main__":
    main()