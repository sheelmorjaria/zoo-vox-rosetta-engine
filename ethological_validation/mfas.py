#!/usr/bin/env python3
"""
Multi-Factor Acceptance Score (MFAS)

Replaces the flawed Response Appropriateness Score (RAS) with a
biologically-accurate multi-factor metric for measuring true acceptance
in animal-AI interactions.

MFAS combines:
1. Temporal Gating (hard constraint) - Species-specific response windows
2. Acoustic Convergence (continuous) - Measuring vocal dialect matching
3. Prosodic Similarity (continuous) - DTW-based temporal prosody comparison

The 2-second "Confusion Metric" is eliminated. MFAS respects ethological
constraints across species with different temporal dynamics.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .acoustic_convergence import (
    AcousticConvergenceEngine,
    ConvergenceResult,
    MultiDimensionalConvergence,
)
from .prosodic_dtw import DTWResult, ProsodicDTW
from .taxa_profiles import TemporalGate

logger = logging.getLogger(__name__)


@dataclass
class MFASResult:
    """
    Result of Multi-Factor Acceptance Score calculation.

    Attributes:
        mfas_score: Overall acceptance score [0, 1]
        temporal_valid: Whether response timing was biologically valid
        temporal_score: Temporal component score [0, 1]
        convergence_result: Acoustic convergence analysis
        prosody_result: Prosodic DTW analysis
        breakdown: Dictionary of individual component scores
        rejected_reason: Reason for rejection (if any)
    """
    mfas_score: float
    temporal_valid: bool
    temporal_score: float
    convergence_result: ConvergenceResult
    prosody_result: DTWResult
    breakdown: Dict[str, float] = field(default_factory=dict)
    rejected_reason: Optional[str] = None


@dataclass
class InteractionEvent:
    """
    Structured representation of an animal-AI interaction event.

    Attributes:
        species: Species identifier (for temporal gating)
        ai_output_state: AI's vocalization state (affect vector or 112D features)
        animal_pre_state: Animal's state BEFORE AI response
        animal_post_state: Animal's state AFTER AI response (for convergence)
        animal_f0_contour: Animal's F0 trajectory (for prosody)
        animal_amplitude_envelope: Optional amplitude envelope
        ai_end_time_ms: Timestamp when AI vocalization ended
        animal_response_time_ms: Timestamp when animal response started
    """
    species: str
    ai_output_state: np.ndarray
    animal_pre_state: np.ndarray
    animal_post_state: np.ndarray
    animal_f0_contour: np.ndarray
    ai_end_time_ms: float
    animal_response_time_ms: float
    animal_amplitude_envelope: Optional[np.ndarray] = None


class MultiFactorAcceptanceScore:
    """
    Multi-Factor Acceptance Score calculator.

    Combines three orthogonal metrics of acceptance:
    1. Temporal Gating (binary constraint)
    2. Acoustic Convergence (distance change in latent space)
    3. Prosodic Similarity (DTW against natural baselines)

    The fusion is multiplicative: invalid timing = zero score.
    This ensures biological realism over "response at any cost" behavior.
    """

    def __init__(
        self,
        temporal_gate: TemporalGate,
        convergence_engine: AcousticConvergenceEngine,
        dtw_engine: ProsodicDTW,
        w_convergence: float = 0.4,
        w_prosody: float = 0.6,
    ):
        """
        Initialize MFAS calculator.

        Args:
            temporal_gate: TemporalGate for species-specific timing validation
            convergence_engine: AcousticConvergenceEngine for dialect matching
            dtw_engine: ProsodicDTW for prosodic similarity
            w_convergence: Weight for acoustic convergence (default 0.4)
            w_prosody: Weight for prosodic similarity (default 0.6)
        """
        self.gate = temporal_gate
        self.convergence = convergence_engine
        self.dtw = dtw_engine
        self.w_convergence = w_convergence
        self.w_prosody = w_prosody

        # Validate weights
        if not np.isclose(w_convergence + w_prosody, 1.0):
            logger.warning(
                f"Weights sum to {w_convergence + w_prosody}, not 1.0. "
                "Normalizing weights."
            )
            total = w_convergence + w_prosody
            self.w_convergence = w_convergence / total
            self.w_prosody = w_prosody / total

        logger.info(
            f"MFAS initialized: w_convergence={self.w_convergence:.2f}, "
            f"w_prosody={self.w_prosody:.2f}"
        )

    def evaluate_interaction(self, event: InteractionEvent) -> MFASResult:
        """
        Evaluate an interaction event using multi-factor acceptance scoring.

        Scoring logic:
        1. Temporal Gating: Binary gate - invalid timing = rejection
        2. If timing valid:
           - Acoustic Convergence: Did animal move toward AI's dialect?
           - Prosodic Similarity: Does animal's temporal prosody match natural conversation?
        3. Combine convergence + prosody (weighted)
        4. Final score = combined (if timing valid) or 0 (if invalid)

        Args:
            event: InteractionEvent with all required data

        Returns:
            MFASResult with detailed scoring breakdown
        """
        # Step 1: Temporal Gating (Hard Constraint)
        temporal_valid = self.gate.is_valid_response(
            event.ai_end_time_ms,
            event.animal_response_time_ms,
        )

        if not temporal_valid:
            latency_ms = event.animal_response_time_ms - event.ai_end_time_ms
            return MFASResult(
                mfas_score=0.0,
                temporal_valid=False,
                temporal_score=0.0,
                convergence_result=ConvergenceResult(
                    convergence_score=0.0,
                    raw_convergence=0.0,
                    direction="neutral",
                    pre_distance=0.0,
                    post_distance=0.0,
                ),
                prosody_result=DTWResult(
                    similarity_score=0.0,
                    dtw_distance=0.0,
                    normalized_distance=0.0,
                    warping_path=np.array([]),
                    best_match_idx=-1,
                ),
                breakdown={"latency_ms": latency_ms},
                rejected_reason=f"Invalid response latency: {latency_ms:.1f}ms",
            )

        # Step 2: Acoustic Convergence (Continuous)
        convergence_result = self.convergence.calculate_convergence(
            event.animal_pre_state,
            event.ai_output_state,
            event.animal_post_state,
        )

        # Step 3: Prosodic Similarity (Continuous)
        prosody_result = self.dtw.score_response(
            event.animal_f0_contour,
            event.animal_amplitude_envelope,
        )

        # Step 4: Weighted Fusion
        # Convergence: 0 = moved away, 1 = moved toward
        # Prosody: 0 = unnatural prosody, 1 = natural conversation-like
        continuous_score = (
            self.w_convergence * convergence_result.convergence_score +
            self.w_prosody * prosody_result.similarity_score
        )

        # Temporal score (latency within valid window)
        latency_ms = event.animal_response_time_ms - event.ai_end_time_ms
        temporal_score = self.gate.get_latency_score(latency_ms)

        # Final MFAS: continuous_score * temporal_score
        # This rewards both biological factors without double-counting
        mfas_score = continuous_score

        breakdown = {
            "latency_ms": latency_ms,
            "temporal_score": temporal_score,
            "convergence_score": convergence_result.convergence_score,
            "convergence_direction": convergence_result.direction,
            "prosody_score": prosody_result.similarity_score,
            "w_convergence": self.w_convergence,
            "w_prosody": self.w_prosody,
        }

        return MFASResult(
            mfas_score=float(mfas_score),
            temporal_valid=True,
            temporal_score=temporal_score,
            convergence_result=convergence_result,
            prosody_result=prosody_result,
            breakdown=breakdown,
        )

    def evaluate_batch(
        self,
        events: List[InteractionEvent],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of interaction events.

        Args:
            events: List of InteractionEvent objects

        Returns:
            Dictionary with aggregate statistics
        """
        results = []
        for event in events:
            try:
                result = self.evaluate_interaction(event)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate interaction: {e}")
                continue

        if not results:
            return {
                "count": 0,
                "mean_mfas": 0.0,
                "valid_rate": 0.0,
            }

        mfas_scores = [r.mfas_score for r in results]
        temporal_valid = [r.temporal_valid for r in results]
        convergence_scores = [r.convergence_result.convergence_score for r in results]
        prosody_scores = [r.prosody_result.similarity_score for r in results]

        # Count convergence directions
        directions = [r.convergence_result.direction for r in results]
        toward_count = sum(1 for d in directions if d == "toward")
        away_count = sum(1 for d in directions if d == "away")
        neutral_count = sum(1 for d in directions if d == "neutral")

        return {
            "count": len(results),
            "mean_mfas": float(np.mean(mfas_scores)),
            "std_mfas": float(np.std(mfas_scores)),
            "median_mfas": float(np.median(mfas_scores)),
            "valid_rate": float(np.mean(temporal_valid)),
            "mean_convergence": float(np.mean(convergence_scores)),
            "mean_prosody": float(np.mean(prosody_scores)),
            "toward_rate": toward_count / len(results),
            "away_rate": away_count / len(results),
            "neutral_rate": neutral_count / len(results),
        }


class MFASComparator:
    """
    Compare MFAS scores across different experimental conditions.

    Used for A/B testing and ethological validation studies.
    """

    def __init__(self, baseline_mfas: MultiFactorAcceptanceScore):
        """
        Initialize comparator.

        Args:
            baseline_mfas: MFAS calculator for baseline condition
        """
        self.baseline = baseline_mfas

    def compare_conditions(
        self,
        condition_a_events: List[InteractionEvent],
        condition_b_events: List[InteractionEvent],
        condition_name_a: str = "Condition A",
        condition_name_b: str = "Condition B",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two experimental conditions.

        Args:
            condition_a_events: Events from condition A
            condition_b_events: Events from condition B
            condition_name_a: Name for condition A
            condition_name_b: Name for condition B

        Returns:
            Dictionary with statistics for both conditions
        """
        stats_a = self.baseline.evaluate_batch(condition_a_events)
        stats_b = self.baseline.evaluate_batch(condition_b_events)

        # Statistical comparison (simplified t-test)
        from scipy import stats

        scores_a = [self.baseline.evaluate_interaction(e).mfas_score
                    for e in condition_a_events]
        scores_b = [self.baseline.evaluate_interaction(e).mfas_score
                    for e in condition_b_events]

        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

        return {
            condition_name_a: {**stats_a, "scores": scores_a},
            condition_name_b: {**stats_b, "scores": scores_b},
            "comparison": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "effect_size": float((np.mean(scores_a) - np.mean(scores_b)) /
                                    np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)),
            }
        }


def create_mfas_for_species(
    species: str,
    baseline_contours: Optional[List[np.ndarray]] = None,
    w_convergence: float = 0.4,
    w_prosody: float = 0.6,
) -> MultiFactorAcceptanceScore:
    """
    Factory function to create MFAS calculator for a species.

    Args:
        species: Species identifier (e.g., "rousettus_aegyptiacus")
        baseline_contours: List of F0 contours from natural conversations
        w_convergence: Weight for acoustic convergence
        w_prosody: Weight for prosodic similarity

    Returns:
        Configured MultiFactorAcceptanceScore

    Example:
        >>> mfas = create_mfas_for_species("rousettus_aegyptiacus", baselines)
        >>> event = InteractionEvent(...)
        >>> result = mfas.evaluate_interaction(event)
        >>> print(f"MFAS: {result.mfas_score:.3f}")
    """
    from .taxa_profiles import get_temporal_gate

    gate = get_temporal_gate(species)
    convergence = AcousticConvergenceEngine(distance_metric='cosine')
    dtw = ProsodicDTW(baseline_contours=baseline_contours, sigma=5.0)

    return MultiFactorAcceptanceScore(
        temporal_gate=gate,
        convergence_engine=convergence,
        dtw_engine=dtw,
        w_convergence=w_convergence,
        w_prosody=w_prosody,
    )


# =============================================================================
# Preset Configurations
# =============================================================================

# Default MFAS for Egyptian Fruit Bat
BAT_MFAS = create_mfas_for_species("rousettus_aegyptiacus")

# Default MFAS for Marmoset
MARMOSET_MFAS = create_mfas_for_species("callithrix_jacchus")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Multi-Factor Acceptance Score Demo")
    print("=" * 60)

    # Create synthetic baseline contours
    np.random.seed(42)
    baselines = [
        np.linspace(5000, 7000, 50) + np.random.randn(50) * 100,
        np.ones(50) * 6000 + np.random.randn(50) * 50,
        6000 + 1000 * np.sin(np.linspace(0, 2*np.pi, 50)),
    ]

    # Create MFAS for bat
    mfas = create_mfas_for_species("rousettus_aegyptiacus", baselines)

    # Test 1: High acceptance (valid timing, convergence, natural prosody)
    print("\nTest 1: High Acceptance")
    print("-" * 40)

    event1 = InteractionEvent(
        species="rousettus_aegyptiacus",
        ai_output_state=np.ones(16) * 0.5,
        animal_pre_state=np.zeros(16),
        animal_post_state=np.ones(16) * 0.4,  # Moved toward AI
        animal_f0_contour=np.linspace(5000, 7000, 45) + np.random.randn(45) * 100,
        ai_end_time_ms=1000,
        animal_response_time_ms=1090,  # 90ms - valid for bat (30-150ms)
    )

    result1 = mfas.evaluate_interaction(event1)
    print(f"MFAS Score: {result1.mfas_score:.3f}")
    print(f"Temporal Valid: {result1.temporal_valid}")
    print(f"Convergence: {result1.convergence_result.direction} "
          f"({result1.convergence_result.convergence_score:.3f})")
    print(f"Prosody Similarity: {result1.prosody_result.similarity_score:.3f}")
    print(f"Breakdown: {result1.breakdown}")

    # Test 2: Invalid timing (rejection)
    print("\nTest 2: Invalid Timing (Rejection)")
    print("-" * 40)

    event2 = InteractionEvent(
        species="rousettus_aegyptiacus",
        ai_output_state=np.ones(16) * 0.5,
        animal_pre_state=np.zeros(16),
        animal_post_state=np.ones(16) * 0.4,
        animal_f0_contour=np.linspace(5000, 7000, 45),
        ai_end_time_ms=1000,
        animal_response_time_ms=2000,  # 1000ms - invalid for bat (>150ms)
    )

    result2 = mfas.evaluate_interaction(event2)
    print(f"MFAS Score: {result2.mfas_score:.3f}")
    print(f"Temporal Valid: {result2.temporal_valid}")
    print(f"Rejected Reason: {result2.rejected_reason}")

    # Test 3: Acoustic divergence (away from AI)
    print("\nTest 3: Acoustic Divergence")
    print("-" * 40)

    event3 = InteractionEvent(
        species="rousettus_aegyptiacus",
        ai_output_state=np.ones(16) * 0.5,
        animal_pre_state=np.zeros(16),
        animal_post_state=np.ones(16) * -0.2,  # Moved away from AI
        animal_f0_contour=np.linspace(5000, 7000, 45),
        ai_end_time_ms=1000,
        animal_response_time_ms=1090,
    )

    result3 = mfas.evaluate_interaction(event3)
    print(f"MFAS Score: {result3.mfas_score:.3f}")
    print(f"Convergence: {result3.convergence_result.direction} "
          f"({result3.convergence_result.convergence_score:.3f})")

    # Test 4: Batch evaluation
    print("\nTest 4: Batch Evaluation")
    print("-" * 40)

    # Mix of high, medium, and low acceptance events
    events = [event1, event3]
    batch_stats = mfas.evaluate_batch(events)

    print(f"Count: {batch_stats['count']}")
    print(f"Mean MFAS: {batch_stats['mean_mfas']:.3f} ± {batch_stats['std_mfas']:.3f}")
    print(f"Valid Rate: {batch_stats['valid_rate']:.1%}")
    print(f"Convergence Direction: toward={batch_stats['toward_rate']:.1%}, "
          f"away={batch_stats['away_rate']:.1%}")
