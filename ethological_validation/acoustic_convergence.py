#!/usr/bin/env python3
"""
Acoustic Convergence Metrics for Ethological Validation

Measures true acceptance: In vocal learning species, acceptance is indicated
by the animal modifying its own vocal parameters to match the AI's output
(vocal convergence or "dialect matching").

This replaces the naive "response rate" metric which cannot distinguish
between aggressive responses (high engagement but negative) and true
conversational acceptance.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Result of acoustic convergence analysis."""
    convergence_score: float  # Normalized [0, 1]
    raw_convergence: float    # Raw distance change
    direction: str            # "toward", "away", or "neutral"
    pre_distance: float       # Distance before AI
    post_distance: float      # Distance after AI


class AcousticConvergenceEngine:
    """
    Measures vocal convergence: does the animal shift its acoustic
    state toward the AI's output?

    In vocal learning species (bats, marmosets, dolphins, songbirds),
    acceptance is indicated by the animal modifying its own vocalization
    to match the interlocutor. Aggression or confusion typically involves
    divergence (moving away in acoustic space) or stereotyped displays.

    The engine operates on the 16D VAE affect vector or full 112D
    RosettaFeatures, computing distances in latent space.
    """

    def __init__(
        self,
        distance_metric: Literal['cosine', 'euclidean', 'mahalanobis'] = 'cosine',
        covariance_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize convergence engine.

        Args:
            distance_metric: Distance metric to use
                - 'cosine': Angular distance (default for VAE latent space)
                - 'euclidean': L2 distance
                - 'mahalanobis': Covariance-weighted distance
            covariance_matrix: Required for Mahalanobis distance (16x16 or 112x112)
        """
        self.distance_metric = distance_metric

        if distance_metric == 'mahalanobis':
            if covariance_matrix is None:
                raise ValueError("covariance_matrix required for Mahalanobis distance")
            self.covariance = covariance_matrix
            self.covariance_inv = np.linalg.inv(covariance_matrix + 1e-6 * np.eye(covariance_matrix.shape[0]))

        logger.info(f"AcousticConvergenceEngine initialized with {distance_metric} distance")

    def calculate_convergence(
        self,
        animal_pre_state: np.ndarray,
        ai_output_state: np.ndarray,
        animal_post_state: np.ndarray,
    ) -> ConvergenceResult:
        """
        Calculate convergence score.

        A positive score means the animal moved toward the AI's dialect
        (acceptance). A negative score means divergence (rejection/aggression).

        Args:
            animal_pre_state: Animal's vocalization BEFORE AI (baseline)
            ai_output_state: AI's synthesized vocalization (target)
            animal_post_state: Animal's response AFTER AI (convergence check)

        Returns:
            ConvergenceResult with normalized score and metrics
        """
        # Compute distances
        dist_pre_ai = self._compute_distance(animal_pre_state, ai_output_state)
        dist_post_ai = self._compute_distance(animal_post_state, ai_output_state)

        # Convergence: positive = moved toward AI, negative = moved away
        raw_convergence = dist_pre_ai - dist_post_ai

        # Determine direction
        if abs(raw_convergence) < 0.01:  # Threshold for "neutral"
            direction = "neutral"
        elif raw_convergence > 0:
            direction = "toward"
        else:
            direction = "away"

        # Normalize to [0, 1]
        # Use sigmoid-like mapping centered at 0
        # Positive convergence -> high score (acceptance)
        # Negative convergence -> low score (rejection)
        convergence_score = 1.0 / (1.0 + np.exp(-10 * raw_convergence))

        return ConvergenceResult(
            convergence_score=float(convergence_score),
            raw_convergence=float(raw_convergence),
            direction=direction,
            pre_distance=float(dist_pre_ai),
            post_distance=float(dist_post_ai),
        )

    def _compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.distance_metric == 'cosine':
            return self._cosine_distance(v1, v2)
        elif self.distance_metric == 'euclidean':
            return self._euclidean_distance(v1, v2)
        elif self.distance_metric == 'mahalanobis':
            return self._mahalanobis_distance(v1, v2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine distance: 1 - cos(angle)."""
        v1_flat = v1.flatten()
        v2_flat = v2.flatten()

        norm1 = np.linalg.norm(v1_flat)
        norm2 = np.linalg.norm(v2_flat)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        cosine_sim = np.dot(v1_flat, v2_flat) / (norm1 * norm2)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

        return 1.0 - cosine_sim

    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """L2 distance."""
        return np.linalg.norm(v1.flatten() - v2.flatten())

    def _mahalanobis_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Mahalanobis distance: sqrt((v1-v2)^T * Sigma^-1 * (v1-v2))."""
        diff = (v1 - v2).flatten()
        return np.sqrt(diff @ self.covariance_inv @ diff)


class MultiDimensionalConvergence:
    """
    Extended convergence analysis across multiple acoustic dimensions.

    Analyzes convergence separately for different acoustic features:
    - F0 (fundamental frequency)
    - Harmonic amplitudes (spectral envelope)
    - Noise characteristics (breathiness)
    - Temporal features (rhythm, duration)
    """

    def __init__(self):
        """Initialize multi-dimensional convergence analyzer."""
        self.dimensions = {
            'f0': lambda state: state[0] if len(state) > 0 else 0.0,
            'harmonics': lambda state: state[1:61] if len(state) >= 61 else np.zeros(60),
            'noise': lambda state: state[61:66] if len(state) >= 66 else np.zeros(5),
            'affect': lambda state: state[-16:] if len(state) >= 16 else np.zeros(16),
        }

    def calculate_dimensional_convergence(
        self,
        animal_pre: np.ndarray,
        ai_output: np.ndarray,
        animal_post: np.ndarray,
    ) -> dict:
        """
        Calculate convergence for each dimension separately.

        Returns:
            Dictionary mapping dimension name to ConvergenceResult
        """
        results = {}

        for dim_name, extractor in self.dimensions.items():
            try:
                pre_val = extractor(animal_pre)
                ai_val = extractor(ai_output)
                post_val = extractor(animal_post)

                # Compute relative change
                if dim_name == 'f0':
                    # Log-scale for F0
                    pre_val = np.log(pre_val + 1)
                    ai_val = np.log(ai_val + 1)
                    post_val = np.log(post_val + 1)

                # Compute distances
                dist_pre = abs(pre_val - ai_val) if np.isscalar(pre_val) else np.linalg.norm(pre_val - ai_val)
                dist_post = abs(post_val - ai_val) if np.isscalar(post_val) else np.linalg.norm(post_val - ai_val)

                convergence = dist_pre - dist_post
                score = 1.0 / (1.0 + np.exp(-10 * convergence))

                results[dim_name] = ConvergenceResult(
                    convergence_score=float(score),
                    raw_convergence=float(convergence),
                    direction="toward" if convergence > 0.01 else "away" if convergence < -0.01 else "neutral",
                    pre_distance=float(dist_pre),
                    post_distance=float(dist_post),
                )
            except Exception as e:
                logger.warning(f"Failed to compute {dim_name} convergence: {e}")
                results[dim_name] = None

        return results


def compute_convergence_from_affect_vectors(
    animal_pre_affect: np.ndarray,  # 16D
    ai_affect: np.ndarray,           # 16D
    animal_post_affect: np.ndarray,  # 16D
) -> float:
    """
    Convenience function for convergence calculation using 16D affect vectors.

    This is the primary interface for Stage 2/3 VAE-based systems.

    Args:
        animal_pre_affect: Animal's affect vector before AI
        ai_affect: AI's generated affect vector
        animal_post_affect: Animal's affect vector after hearing AI

    Returns:
        Convergence score [0, 1] where higher = more acceptance

    Example:
        >>> # Animal starts neutral, hears AI, moves toward AI's state
        >>> pre = np.zeros(16)
        >>> ai = np.ones(16) * 0.5  # Moderate arousal
        >>> post = np.ones(16) * 0.4  # Moved toward AI
        >>> score = compute_convergence_from_affect_vectors(pre, ai, post)
        >>> print(f"Convergence: {score:.3f}")
    """
    engine = AcousticConvergenceEngine(distance_metric='cosine')
    result = engine.calculate_convergence(animal_pre_affect, ai_affect, animal_post_affect)
    return result.convergence_score


def compute_batch_convergence(
    batch_interactions: list[dict],
    distance_metric: str = 'cosine',
) -> dict:
    """
    Compute convergence scores for a batch of interactions.

    Args:
        batch_interactions: List of dicts with 'animal_pre', 'ai_output', 'animal_post'
        distance_metric: Distance metric to use

    Returns:
        Dictionary with aggregate statistics
    """
    engine = AcousticConvergenceEngine(distance_metric=distance_metric)

    scores = []
    raw_convergences = []
    directions = {'toward': 0, 'away': 0, 'neutral': 0}

    for interaction in batch_interactions:
        try:
            result = engine.calculate_convergence(
                interaction['animal_pre'],
                interaction['ai_output'],
                interaction['animal_post'],
            )
            scores.append(result.convergence_score)
            raw_convergences.append(result.raw_convergence)
            directions[result.direction] += 1
        except Exception as e:
            logger.warning(f"Failed to compute convergence: {e}")
            continue

    if not scores:
        return {'count': 0, 'mean_score': 0, 'toward_rate': 0}

    import numpy as np

    return {
        'count': len(scores),
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'median_score': float(np.median(scores)),
        'mean_raw_convergence': float(np.mean(raw_convergences)),
        'toward_rate': directions['toward'] / len(scores),
        'away_rate': directions['away'] / len(scores),
        'neutral_rate': directions['neutral'] / len(scores),
    }


# =============================================================================
# Preset Configurations
# =============================================================================

# Default convergence engine for 16D VAE affect space
DEFAULT_CONVERGENCE_ENGINE = AcousticConvergenceEngine(
    distance_metric='cosine'
)

# Multi-dimensional analyzer
MULTI_DIM_CONVERGENCE = MultiDimensionalConvergence()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Acoustic Convergence Engine Demo")
    print("=" * 50)

    # Simulate interaction
    animal_pre = np.array([0.0] * 16)  # Neutral start
    ai_output = np.array([0.5] * 16)   # AI outputs moderate arousal
    animal_post = np.array([0.4] * 16)  # Animal moves toward AI (acceptance)

    result = DEFAULT_CONVERGENCE_ENGINE.calculate_convergence(
        animal_pre, ai_output, animal_post
    )

    print(f"Convergence Score: {result.convergence_score:.4f}")
    print(f"Direction: {result.direction}")
    print(f"Raw Convergence: {result.raw_convergence:.4f}")

    # Test multi-dimensional analysis
    full_state_112d = np.random.randn(112)
    full_state_2 = np.random.randn(112) * 0.5 + full_state_112d

    results = MULTI_DIM_CONVERGENCE.calculate_dimensional_convergence(
        full_state_112d,
        full_state_2,
        full_state_112d * 0.9 + full_state_2 * 0.1,  # Slight convergence
    )

    print("\nMulti-Dimensional Convergence:")
    for dim, res in results.items():
        if res:
            print(f"  {dim}: {res.direction} (score: {res.convergence_score:.3f})")
