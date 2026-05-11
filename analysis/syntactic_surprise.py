#!/usr/bin/env python3
"""
Predictive Syntactic "Surprise" Analysis

Uses the Autoregressive Transformer's probability outputs to calculate
the "Surprise" (negative log-likelihood) of a bat's subsequent vocalization.

Measures information-theoretic surprise - how unexpected was a bat's
response given the syntactic context? High surprise indicates:
- Rule-breaking innovation
- Context switching
- Potential deception attempts

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SurpriseEvent:
    """
    A single surprise measurement event.

    Attributes:
        sequence_id: Unique identifier
        context_tokens: Preceding token sequence
        actual_token: The token that actually occurred
        predicted_probs: Model's probability distribution over vocabulary
        surprise: Negative log-likelihood (bits)
        entropy: Entropy of prediction distribution
        rank: Rank of actual token in predicted ordering
    """
    sequence_id: str
    context_tokens: Tuple[int, ...]
    actual_token: int
    predicted_probs: np.ndarray  # (vocab_size,)
    surprise: float  # -log(p(actual_token))
    entropy: float  # Shannon entropy of prediction
    rank: int  # 1 = most probable, vocab_size = least


@dataclass
class SurpriseProfile:
    """
    Surprise profile for an individual bat.

    Tracks baseline surprise, surprise bursts, and
    innovation patterns.
    """
    bat_id: int
    mean_surprise: float
    std_surprise: float
    median_surprise: float
    surprise_bursts: List[SurpriseEvent]  # Events > 2 std from mean
    innovation_events: List[SurpriseEvent]  # Very high surprise
    conformity_events: List[SurpriseEvent]  # Very low surprise (predictable)

    # Context-dependent surprise
    context_surprise: Dict[Tuple[int, ...], float]  # context -> mean surprise


class SyntacticSurpriseAnalyzer:
    """
    Analyzes information-theoretic surprise in bat vocalizations.

    Uses autoregressive transformer to compute the probability
    of each observed token given its context, then calculates
    surprise as negative log-likelihood.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        surprise_threshold: float = 2.0,  # std devs for "burst"
        innovation_threshold: float = 3.0,  # std devs for "innovation"
    ):
        """
        Initialize syntactic surprise analyzer.

        Args:
            vocab_size: Size of VQ-VAE codebook
            surprise_threshold: std devs above mean for "surprise burst"
            innovation_threshold: std devs above mean for "innovation"
        """
        self.vocab_size = vocab_size
        self.surprise_threshold = surprise_threshold
        self.innovation_threshold = innovation_threshold

        # Per-bat profiles
        self.profiles: Dict[int, SurpriseProfile] = {}

        # Baseline surprise (from corpus)
        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0

        logger.info("SyntacticSurpriseAnalyzer initialized")

    def compute_surprise(
        self,
        context_tokens: Tuple[int, ...],
        actual_token: int,
        predicted_probs: np.ndarray,
    ) -> SurpriseEvent:
        """
        Compute surprise for a single token.

        Args:
            context_tokens: Preceding tokens
            actual_token: The token that occurred
            predicted_probs: Model's P(vocab | context)

        Returns:
            SurpriseEvent with computed metrics
        """
        # Probability of actual token
        p_actual = predicted_probs[actual_token]

        # Surprise = -log2(p) in bits (base-2 for information theory)
        surprise = -np.log2(p_actual + 1e-10)

        # Entropy of prediction distribution
        # H = -sum(p * log2(p))
        entropy = -np.sum(
            predicted_probs * np.log2(predicted_probs + 1e-10)
        )

        # Rank of actual token (1-indexed)
        rank = np.argsort(np.argsort(-predicted_probs))[actual_token] + 1

        return SurpriseEvent(
            sequence_id="",
            context_tokens=context_tokens,
            actual_token=actual_token,
            predicted_probs=predicted_probs,
            surprise=surprise,
            entropy=entropy,
            rank=rank,
        )

    def analyze_sequence_surprise(
        self,
        token_sequence: List[int],
        model_predictions: List[np.ndarray],
        sequence_id: str = "",
    ) -> List[SurpriseEvent]:
        """
        Analyze surprise across a full token sequence.

        Args:
            token_sequence: Full sequence of tokens
            model_predictions: List of probability distributions,
                              one per position (skip first)
            sequence_id: Unique identifier

        Returns:
            List of SurpriseEvents for each position
        """
        events = []

        for i in range(1, len(token_sequence)):
            context = tuple(token_sequence[:i])
            actual = token_sequence[i]

            if i - 1 < len(model_predictions):
                event = self.compute_surprise(
                    context, actual, model_predictions[i - 1]
                )
                event.sequence_id = sequence_id
                events.append(event)

        return events

    def compute_surprise_profile(
        self,
        bat_id: int,
        events: List[SurpriseEvent],
    ) -> SurpriseProfile:
        """
        Compute comprehensive surprise profile for a bat.

        Args:
            bat_id: Bat identifier
            events: List of surprise events for this bat

        Returns:
            SurpriseProfile with statistics
        """
        if not events:
            return SurpriseProfile(
                bat_id=bat_id,
                mean_surprise=0.0,
                std_surprise=0.0,
                median_surprise=0.0,
                surprise_bursts=[],
                innovation_events=[],
                conformity_events=[],
                context_surprise={},
            )

        surprises = [e.surprise for e in events]
        mean_surprise = np.mean(surprises)
        std_surprise = np.std(surprises)
        median_surprise = np.median(surprises)

        # Classify events
        surprise_bursts = []
        innovation_events = []
        conformity_events = []

        for event in events:
            z_score = (event.surprise - mean_surprise) / (std_surprise + 1e-10)

            if z_score > self.innovation_threshold:
                innovation_events.append(event)
            elif z_score > self.surprise_threshold:
                surprise_bursts.append(event)
            elif z_score < -self.surprise_threshold:
                # Unusually predictable
                conformity_events.append(event)

        # Context-dependent surprise
        context_surprise = {}
        context_groups: Dict[Tuple[int, ...], List[float]] = {}

        for event in events:
            if event.context_tokens not in context_groups:
                context_groups[event.context_tokens] = []
            context_groups[event.context_tokens].append(event.surprise)

        for ctx, surs in context_groups.items():
            context_surprise[ctx] = np.mean(surs)

        return SurpriseProfile(
            bat_id=bat_id,
            mean_surprise=mean_surprise,
            std_surprise=std_surprise,
            median_surprise=median_surprise,
            surprise_bursts=surprise_bursts,
            innovation_events=innovation_events,
            conformity_events=conformity_events,
            context_surprise=context_surprise,
        )

    def detect_deception_candidates(
        self,
        profile: SurpriseProfile,
    ) -> List[SurpriseEvent]:
        """
        Detect potential deception attempts.

        Hypothesis: Deceptive calls show anomalous surprise patterns -
        very high surprise (rule-breaking) but delivered with
        low-arousal affect (calm delivery).

        Args:
            profile: Surprise profile to analyze

        Returns:
            List of candidate deception events
        """
        # This would need affect data integration
        # For now, return innovation events as candidates
        return profile.innovation_events

    def compare_surprise_profiles(
        self,
        profile1: SurpriseProfile,
        profile2: SurpriseProfile,
    ) -> Dict:
        """
        Statistically compare two bats' surprise profiles.

        Tests if one bat is significantly more surprising/innovative
        than another.

        Args:
            profile1: First bat's profile
            profile2: Second bat's profile

        Returns:
            Dictionary with statistical comparison
        """
        # Mann-Whitney U test for distribution comparison
        # (non-parametric, robust to outliers)

        # Would need full event lists for this
        # Simplified: compare means

        mean_diff = profile1.mean_surprise - profile2.mean_surprise

        # Pooled std
        pooled_std = np.sqrt(
            (profile1.std_surprise ** 2 + profile2.std_surprise ** 2) / 2
        )

        # Effect size (Cohen's d)
        effect_size = mean_diff / (pooled_std + 1e-10)

        return {
            "mean_difference": mean_diff,
            "effect_size": effect_size,
            "profile1_more_surprising": mean_diff > 0,
            "innovation_count_diff": (
                len(profile1.innovation_events) -
                len(profile2.innovation_events)
            ),
        }

    def track_surprise_over_time(
        self,
        events: List[SurpriseEvent],
        window_ms: float = 300000,  # 5 minutes
    ) -> List[Tuple[float, float]]:
        """
        Track surprise evolution over time.

        Uses sliding window to detect trends like:
- Increasing surprise (innovation period)
- Decreasing surprise (conventionalization)

        Args:
            events: Surprise events with timestamps
            window_ms: Sliding window duration

        Returns:
            List of (timestamp, mean_surprise) tuples
        """
        # This would need timestamps in SurpriseEvent
        # For now, return empty
        return []

    def test_surprise_hypothesis(
        self,
        surprise_values: List[float],
        baseline_mean: float,
        baseline_std: float,
    ) -> Tuple[bool, float, str]:
        """
        Test if surprise values differ significantly from baseline.

        One-sample t-test comparing observed surprise to baseline.

        Args:
            surprise_values: Observed surprise values
            baseline_mean: Baseline mean surprise
            baseline_std: Baseline std surprise

        Returns:
            (significant, p_value, interpretation) tuple
        """
        if len(surprise_values) < 10:
            return False, 1.0, "Insufficient data"

        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(
            surprise_values,
            baseline_mean
        )

        # Effect size (Cohen's d)
        effect_size = (
            (np.mean(surprise_values) - baseline_mean) /
            (baseline_std + 1e-10)
        )

        alpha = 0.05
        significant = p_value < alpha

        if significant:
            if effect_size > 0:
                interpretation = (
                    f"Significantly higher surprise (t={t_stat:.2f}, "
                    f"p={p_value:.4f}, d={effect_size:.2f}). "
                    f"Bat is more innovative than baseline."
                )
            else:
                interpretation = (
                    f"Significantly lower surprise (t={t_stat:.2f}, "
                    f"p={p_value:.4f}, d={effect_size:.2f}). "
                    f"Bat is more conventional than baseline."
                )
        else:
            interpretation = (
                f"No significant difference from baseline "
                f"(t={t_stat:.2f}, p={p_value:.4f})."
            )

        return significant, p_value, interpretation


# Preset configurations

# Default syntactic surprise analyzer
DEFAULT_SYNTACTIC_SURPRISE = SyntacticSurpriseAnalyzer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Syntactic Surprise Analysis Demo")
    print("=" * 50)

    analyzer = DEFAULT_SYNTACTIC_SURPRISE

    # Simulate a token sequence
    token_sequence = [5, 10, 5, 10, 5, 10, 42]  # Pattern, then break

    # Simulate model predictions
    vocab_size = 64

    # For position 1: after [5], predict 10
    pred1 = np.zeros(vocab_size)
    pred1[10] = 0.7  # High probability
    pred1[5] = 0.2
    pred1[:] += 0.1 / vocab_size  # Distribute remainder
    pred1 = pred1 / pred1.sum()

    # Position 2: after [5, 10], predict 5
    pred2 = np.zeros(vocab_size)
    pred2[5] = 0.8
    pred2[:] += 0.2 / vocab_size
    pred2 = pred2 / pred2.sum()

    # Position 3: after [5, 10, 5], predict 10
    pred3 = np.zeros(vocab_size)
    pred3[10] = 0.75
    pred3[:] += 0.25 / vocab_size
    pred3 = pred3 / pred3.sum()

    # Position 4: after [5, 10, 5, 10], predict 5
    pred4 = np.zeros(vocab_size)
    pred4[5] = 0.85
    pred4[:] += 0.15 / vocab_size
    pred4 = pred4 / pred4.sum()

    # Position 5: after [5, 10, 5, 10, 5], predict 10
    pred5 = np.zeros(vocab_size)
    pred5[10] = 0.9
    pred5[:] += 0.1 / vocab_size
    pred5 = pred5 / pred5.sum()

    # Position 6: after [5, 10, 5, 10, 5, 10], PREDICT 5, but...
    pred6 = np.zeros(vocab_size)
    pred6[5] = 0.95  # Very confident
    pred6[:] += 0.05 / vocab_size
    pred6 = pred6 / pred6.sum()

    model_predictions = [pred1, pred2, pred3, pred4, pred5, pred6]

    # Analyze
    events = analyzer.analyze_sequence_surprise(
        token_sequence,
        model_predictions,
        sequence_id="demo_seq_001",
    )

    print(f"\nSequence: {token_sequence}")
    print(f"\nSurprise Analysis:")

    for i, event in enumerate(events):
        print(f"  Position {i+1}:")
        print(f"    Context: {event.context_tokens}")
        print(f"    Actual: {event.actual_token}")
        print(f"    Surprise: {event.surprise:.2f} bits")
        print(f"    Rank: {event.rank}/{vocab_size}")

        if event.surprise > 5:
            print(f"    ⚠️  HIGH SURPRISE - Rule breaker!")

    # Compute profile
    profile = analyzer.compute_surprise_profile(bat_id=1, events=events)

    print(f"\nSurprise Profile (Bat 1):")
    print(f"  Mean Surprise: {profile.mean_surprise:.2f} bits")
    print(f"  Std Surprise: {profile.std_surprise:.2f} bits")
    print(f"  Median Surprise: {profile.median_surprise:.2f} bits")
    print(f"  Innovation Events: {len(profile.innovation_events)}")
    print(f"  Conformity Events: {len(profile.conformity_events)}")
