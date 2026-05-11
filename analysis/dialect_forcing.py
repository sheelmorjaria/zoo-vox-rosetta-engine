#!/usr/bin/env python3
"""
Active "Dialect Forcing" Experiment Protocol

Tests vocal learning in real-time by having the AI respond with
smoothly interpolated dialect shifts, measuring if bats converge
their own vocalizations toward the injected dialect.

Uses the continuous VAE latent space to perform Latent-Space
Interpolation, enabling active testing of the "crowd-based
vocal learning" hypothesis.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ethological_validation import (
    AcousticConvergenceEngine,
    ConvergenceResult,
    InteractionEvent,
    MultiFactorAcceptanceScore,
)

logger = logging.getLogger(__name__)


class DialectType(Enum):
    """Predefined dialect types for forcing."""
    DIALECT_A = "dialect_a"  # Low F0, tonal, high HNR
    DIALECT_B = "dialect_b"  # High F0, harsh, low HNR
    DIALECT_C = "dialect_c"  # Intermediate, balanced
    NATURAL = "natural"      # Colony baseline


@dataclass
class DialectDefinition:
    """
    Definition of a dialect in VAE latent space.

    Defines the affect vector characteristics that correspond
    to a specific dialect pattern.
    """
    name: str
    affect_vector: np.ndarray  # 16D prototype
    f0_range: Tuple[float, float]  # (min, max) in Hz
    hnr_range: Tuple[float, float]  # Harmonics-to-noise ratio
    spectral_tilt: float  # -1 to 1, spectral slope
    description: str


@dataclass
class DialectForcingTrial:
    """
    A single dialect forcing trial.

    Records the AI's dialect injection and the colony's response.
    """
    trial_id: str
    bat_id: int
    timestamp_ms: float

    # AI output
    source_dialect: DialectType
    target_dialect: DialectType
    interpolation_factor: float  # 0 = source, 1 = target
    ai_affect_vector: np.ndarray

    # Bat response
    bat_pre_affect: np.ndarray
    bat_post_affect: np.ndarray

    # Convergence metrics
    convergence_result: ConvergenceResult
    mfas_score: float

    # Did bat converge toward target dialect?
    converged: bool


class DialectForcer:
    """
    Performs active dialect forcing experiments.

    Interpolates between dialect prototypes in VAE latent space
    and measures whether bats converge their vocalizations
    toward the forced dialect.
    """

    # Predefined dialect definitions for Egyptian Fruit Bats
    DIALECTS: Dict[DialectType, DialectDefinition] = {
        DialectType.DIALECT_A: DialectDefinition(
            name="Dialect A (Low, Tonal)",
            affect_vector=np.array([
                0.2,   # Low arousal
                -0.3,  # Negative valence (calm)
                0.8,   # High HNR (tonal)
                -0.5,  # Low harshness
                0.1,   # Low jitter
                0.0,   # Neutral
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]),
            f0_range=(5000, 7000),
            hnr_range=(20, 40),
            spectral_tilt=-0.3,
            description="Low-pitched, tonal contact calls",
        ),
        DialectType.DIALECT_B: DialectDefinition(
            name="Dialect B (High, Harsh)",
            affect_vector=np.array([
                0.8,   # High arousal
                0.5,   # Positive valence (excited)
                -0.5,  # Low HNR (noisy)
                0.8,   # High harshness
                0.6,   # High jitter
                0.0,   # Neutral
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]),
            f0_range=(8000, 12000),
            hnr_range=(0, 10),
            spectral_tilt=0.5,
            description="High-pitched, harsh alarm calls",
        ),
        DialectType.DIALECT_C: DialectDefinition(
            name="Dialect C (Intermediate)",
            affect_vector=np.array([
                0.5,   # Medium arousal
                0.0,   # Neutral valence
                0.0,   # Medium HNR
                0.0,   # Medium harshness
                0.3,   # Medium jitter
                0.0,   # Neutral
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]),
            f0_range=(6000, 9000),
            hnr_range=(10, 25),
            spectral_tilt=0.0,
            description="Intermediate social calls",
        ),
        DialectType.NATURAL: DialectDefinition(
            name="Natural Colony Baseline",
            affect_vector=np.zeros(16),
            f0_range=(6000, 8000),
            hnr_range=(15, 30),
            spectral_tilt=0.0,
            description="Natural colony dialect (control)",
        ),
    }

    def __init__(
        self,
        convergence_engine: AcousticConvergenceEngine,
        mfas: MultiFactorAcceptanceScore,
    ):
        """
        Initialize dialect forcing experiment.

        Args:
            convergence_engine: For measuring vocal convergence
            mfas: For measuring acceptance
        """
        self.convergence = convergence_engine
        self.mfas = mfas

        self.trials: List[DialectForcingTrial] = []
        self.baselines: Dict[int, np.ndarray] = {}  # Baseline per bat

        logger.info("DialectForcer initialized")

    def interpolate_dialect(
        self,
        source: DialectType,
        target: DialectType,
        factor: float,
    ) -> np.ndarray:
        """
        Interpolate between two dialects in VAE latent space.

        Args:
            source: Source dialect
            target: Target dialect
            factor: Interpolation factor (0=source, 1=target)

        Returns:
            Interpolated affect vector (16D)
        """
        source_vec = self.DIALECTS[source].affect_vector
        target_vec = self.DIALECTS[target].affect_vector

        # Spherical linear interpolation (SLERP-like)
        # For simplicity, use linear interpolation in affect space
        interpolated = (1 - factor) * source_vec + factor * target_vec

        return interpolated

    def run_forcing_trial(
        self,
        bat_id: int,
        bat_pre_affect: np.ndarray,
        bat_post_affect: np.ndarray,
        source_dialect: DialectType,
        target_dialect: DialectType,
        interpolation_factor: float,
        f0_contour: np.ndarray,
        ai_end_time_ms: float,
        bat_response_time_ms: float,
    ) -> DialectForcingTrial:
        """
        Run a single dialect forcing trial.

        Args:
            bat_id: Bat being tested
            bat_pre_affect: Bat's affect before AI
            bat_post_affect: Bat's affect after AI
            source_dialect: Starting dialect
            target_dialect: Target dialect to force
            interpolation_factor: How far toward target
            f0_contour: Bat's response F0 (for MFAS)
            ai_end_time_ms: AI playback end time
            bat_response_time_ms: Bat response start time

        Returns:
            DialectForcingTrial with results
        """
        # Generate AI output with forced dialect
        ai_affect = self.interpolate_dialect(
            source_dialect,
            target_dialect,
            interpolation_factor,
        )

        # Measure convergence
        convergence = self.convergence.calculate_convergence(
            bat_pre_affect,
            ai_affect,
            bat_post_affect,
        )

        # Compute MFAS
        event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=ai_affect,
            animal_pre_state=bat_pre_affect,
            animal_post_state=bat_post_affect,
            animal_f0_contour=f0_contour,
            ai_end_time_ms=ai_end_time_ms,
            animal_response_time_ms=bat_response_time_ms,
        )
        mfas_result = self.mfas.evaluate_interaction(event)

        # Determine if bat converged
        converged = (
            convergence.direction == "toward" and
            convergence.convergence_score > 0.6
        )

        trial = DialectForcingTrial(
            trial_id=f"trial_{len(self.trials)}",
            bat_id=bat_id,
            timestamp_ms=ai_end_time_ms,
            source_dialect=source_dialect,
            target_dialect=target_dialect,
            interpolation_factor=interpolation_factor,
            ai_affect_vector=ai_affect,
            bat_pre_affect=bat_pre_affect,
            bat_post_affect=bat_post_affect,
            convergence_result=convergence,
            mfas_score=mfas_result.mfas_score,
            converged=converged,
        )

        self.trials.append(trial)
        return trial

    def analyze_forcing_results(
        self,
    ) -> Dict:
        """
        Analyze results across all forcing trials.

        Returns:
            Dictionary with statistics and findings
        """
        if not self.trials:
            return {"error": "No trials run"}

        # Convergence by dialect transition
        transitions = {}
        for trial in self.trials:
            key = (trial.source_dialect, trial.target_dialect)
            if key not in transitions:
                transitions[key] = {"converged": 0, "total": 0}
            transitions[key]["total"] += 1
            if trial.converged:
                transitions[key]["converged"] += 1

        # Convergence by interpolation factor
        factors = {}
        for trial in self.trials:
            factor_bin = int(trial.interpolation_factor * 10) / 10
            if factor_bin not in factors:
                factors[factor_bin] = {"converged": 0, "total": 0}
            factors[factor_bin]["total"] += 1
            if trial.converged:
                factors[factor_bin]["converged"] += 1

        # Overall statistics
        converged_count = sum(1 for t in self.trials if t.converged)
        mean_mfas = np.mean([t.mfas_score for t in self.trials])
        mean_convergence = np.mean([
            t.convergence_result.convergence_score
            for t in self.trials
        ])

        return {
            "total_trials": len(self.trials),
            "converged_count": converged_count,
            "convergence_rate": converged_count / len(self.trials),
            "mean_mfas": mean_mfas,
            "mean_convergence_score": mean_convergence,
            "by_transition": transitions,
            "by_interpolation_factor": factors,
        }

    def test_vocal_learning_hypothesis(
        self,
        alpha: float = 0.05,
    ) -> Tuple[bool, str]:
        """
        Test if bats show vocal learning (convergence to forced dialect).

        Uses statistical test to determine if convergence rate
        is significantly above chance.

        Args:
            alpha: Significance threshold

        Returns:
            (significant, interpretation) tuple
        """
        results = self.analyze_forcing_results()

        if "error" in results:
            return False, "No data available"

        convergence_rate = results["convergence_rate"]

        # Null hypothesis: convergence at chance (50%)
        # Binomial test
        from scipy import stats

        n = results["total_trials"]
        k = results["converged_count"]

        if n < 10:
            return False, f"Insufficient data (n={n})"

        # One-sided binomial test: is convergence > chance?
        p_value = stats.binom_test(k, n, p=0.5, alternative="greater")

        significant = p_value < alpha

        if significant:
            interpretation = (
                f"Vocal learning detected (p={p_value:.4f}). "
                f"Bats converged in {k}/{n} trials ({convergence_rate:.1%})."
            )
        else:
            interpretation = (
                f"No significant vocal learning (p={p_value:.4f}). "
                f"Convergence rate {convergence_rate:.1%} not above chance."
            )

        return significant, interpretation


# Preset configurations

# Default dialect forcing experiment
DEFAULT_DIALECT_FORCER = DialectForcer(
    convergence_engine=AcousticConvergenceEngine(distance_metric='cosine'),
    mfas=None,  # Would be set at runtime
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Dialect Forcing Experiment Demo")
    print("=" * 50)

    # Test interpolation
    dialect_a = DialectForcer.DIALECTS[DialectType.DIALECT_A]
    dialect_b = DialectForcer.DIALECTS[DialectType.DIALECT_B]

    print(f"\nDialect A: {dialect_a.description}")
    print(f"  F0 range: {dialect_a.f0_range}")
    print(f"  HNR range: {dialect_a.hnr_range}")

    print(f"\nDialect B: {dialect_b.description}")
    print(f"  F0 range: {dialect_b.f0_range}")
    print(f"  HNR range: {dialect_b.hnr_range}")

    # Test interpolation
    forcer = DEFAULT_DIALECT_FORCER

    print(f"\nInterpolation Test:")
    for factor in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interpolated = forcer.interpolate_dialect(
            DialectType.DIALECT_A,
            DialectType.DIALECT_B,
            factor,
        )
        print(f"  Factor {factor:.2f}: "
              f"arousal={interpolated[0]:.2f}, "
              f"harshness={interpolated[3]:.2f}")
