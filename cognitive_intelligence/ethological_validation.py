#!/usr/bin/env python3
"""
Ethological Validation Tests (Sprint 5)

Semantic vs Affective Mismatch Test

Tests whether subjects respond differently to Stream 1 (Affect) vs Stream 2 (Syntax)
manipulations through three conditions:

Condition A (Congruent): Matching affect + matching syntax → High response
Condition B (Syntactic Mismatch): Matching affect + mismatched syntax → Low response
Condition C (Affective Mismatch): Mismatched affect + matching syntax → Low response

Measures:
- RAS (Response Appropriateness Score)
- Acoustic Convergence
- Response Latency

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TestCondition(Enum):
    """Test condition for ethological validation."""
    CONGRUENT = "A"  # Matching affect + matching syntax
    SYNTACTIC_MISMATCH = "B"  # Matching affect + mismatched syntax
    AFFECTIVE_MISMATCH = "C"  # Mismatched affect + matching syntax


@dataclass
class TestTrial:
    """Single trial in ethological validation."""
    condition: TestCondition
    stimulus_affect: np.ndarray  # 16D affect vector
    stimulus_token: int  # Syntactic token
    subject_response_affect: Optional[np.ndarray] = None
    subject_response_token: Optional[int] = None
    response_latency_ms: Optional[float] = None
    responded: bool = False
    response_appropriate: bool = False  # Human scoring or automated metric


@dataclass
class SubjectMetrics:
    """Metrics for a single subject."""
    ras_scores: Dict[TestCondition, float]  # Response Appropriateness Score
    acoustic_convergence: Dict[TestCondition, float]  # Acoustic similarity
    response_latencies: Dict[TestCondition, List[float]]  # Response times


class ResponseAppropriatenessScorer:
    """
    Score response appropriateness based on dual-stream congruence.

    Higher score = more appropriate response.
    """

    def __init__(
        self,
        affect_weight: float = 0.5,
        syntax_weight: float = 0.5,
    ):
        self.affect_weight = affect_weight
        self.syntax_weight = syntax_weight

    def score_affect_congruence(
        self,
        stimulus_affect: np.ndarray,
        response_affect: np.ndarray,
    ) -> float:
        """
        Score affective congruence between stimulus and response.

        Higher = more congruent (appropriate matching).
        """
        # Cosine similarity (normalized dot product)
        norm_stim = np.linalg.norm(stimulus_affect)
        norm_resp = np.linalg.norm(response_affect)

        if norm_stim < 1e-8 or norm_resp < 1e-8:
            return 0.0

        similarity = np.dot(stimulus_affect, response_affect) / (norm_stim * norm_resp)

        # Map [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def score_syntax_congruence(
        self,
        stimulus_token: int,
        response_token: int,
        syntax_graph: Any,  # SyntaxGraph with transition probabilities
    ) -> float:
        """
        Score syntactic congruence using transition probability.

        Higher = more probable (appropriate).
        """
        try:
            prob = syntax_graph.get_transition_probability(stimulus_token, response_token)
        except:
            prob = 0.01  # Small default probability

        # Log probability for better discrimination
        return min(1.0, prob * 10)  # Scale up for better range

    def score(
        self,
        stimulus_affect: np.ndarray,
        stimulus_token: int,
        response_affect: np.ndarray,
        response_token: int,
        syntax_graph: Any,
    ) -> float:
        """
        Compute overall response appropriateness score.

        Returns:
            Score between 0 (inappropriate) and 1 (appropriate)
        """
        affect_score = self.score_affect_congruence(stimulus_affect, response_affect)
        syntax_score = self.score_syntax_congruence(stimulus_token, response_token, syntax_graph)

        return (
            self.affect_weight * affect_score +
            self.syntax_weight * syntax_score
        )


class AcousticConvergenceMetric:
    """
    Measure acoustic convergence between stimulus and response.

    Convergence = similarity in acoustic features over time.
    """

    def __init__(self, feature_dim: int = 112):
        self.feature_dim = feature_dim

    def compute_convergence(
        self,
        stimulus_features: np.ndarray,
        response_features: np.ndarray,
    ) -> float:
        """
        Compute acoustic convergence.

        Args:
            stimulus_features: Stimulus acoustic features (112D)
            response_features: Response acoustic features (112D)

        Returns:
            Convergence score (0-1, higher = more converged)
        """
        # Normalize features
        stim_norm = stimulus_features / (np.linalg.norm(stimulus_features) + 1e-8)
        resp_norm = response_features / (np.linalg.norm(response_features) + 1e-8)

        # Cosine similarity
        similarity = np.dot(stim_norm, resp_norm)

        # Map [-1, 1] to [0, 1]
        return (similarity + 1) / 2


class EthologicalValidator:
    """
    Main validator for ethological testing.

    Runs Conditions A, B, C and computes statistical significance.
    """

    def __init__(
        self,
        syntax_graph: Any,
        num_subjects: int = 10,
        num_trials_per_condition: int = 20,
    ):
        self.syntax_graph = syntax_graph
        self.num_subjects = num_subjects
        self.num_trials_per_condition = num_trials_per_condition

        self.scorer = ResponseAppropriatenessScorer()
        self.convergence_metric = AcousticConvergenceMetric()

        # Trials storage
        self.trials: List[TestTrial] = []

        # Results per subject
        self.subject_metrics: Dict[int, SubjectMetrics] = {}

        logger.info(
            f"EthologicalValidator initialized: "
            f"{num_subjects} subjects, {num_trials_per_condition} trials/condition"
        )

    def create_stimulus(
        self,
        condition: TestCondition,
        base_affect: np.ndarray,
        base_token: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Create stimulus for a given condition.

        Returns:
            (stimulus_affect, stimulus_token)
        """
        if condition == TestCondition.CONGRUENT:
            # Matching affect + matching syntax (high probability transition)
            # Find high-probability next token
            valid_next = self.syntax_graph.get_valid_next_tokens(base_token, top_k=3)
            target_token = valid_next[0][0]
            target_affect = base_affect.copy()

        elif condition == TestCondition.SYNTACTIC_MISMATCH:
            # Matching affect + mismatched syntax (low probability transition)
            # Find low-probability next token
            valid_next = self.syntax_graph.get_valid_next_tokens(base_token, top_k=64)
            target_token = valid_next[-1][0]  # Lowest probability
            target_affect = base_affect.copy()

        elif condition == TestCondition.AFFECTIVE_MISMATCH:
            # Mismatched affect + matching syntax
            valid_next = self.syntax_graph.get_valid_next_tokens(base_token, top_k=3)
            target_token = valid_next[0][0]  # High probability
            # Invert arousal for affective mismatch
            target_affect = base_affect.copy()
            target_affect[0] = 1.0 - target_affect[0]  # Invert arousal

        return target_affect, target_token

    def generate_trial_design(
        self,
        base_affects: List[np.ndarray],
        base_tokens: List[int],
    ) -> List[TestTrial]:
        """
        Generate full trial design for all conditions.

        Returns:
            List of trials to run
        """
        trials = []

        for condition in TestCondition:
            for i in range(self.num_trials_per_condition):
                # Select random base stimulus
                base_idx = np.random.randint(len(base_affects))
                base_affect = base_affects[base_idx]
                base_token = base_tokens[base_idx]

                # Create stimulus
                stimulus_affect, stimulus_token = self.create_stimulus(
                    condition, base_affect, base_token
                )

                trial = TestTrial(
                    condition=condition,
                    stimulus_affect=stimulus_affect,
                    stimulus_token=stimulus_token,
                )
                trials.append(trial)

        logger.info(f"Generated {len(trials)} trials for ethological validation")

        return trials

    def record_response(
        self,
        trial: TestTrial,
        response_affect: np.ndarray,
        response_token: int,
        response_latency_ms: float,
    ) -> TestTrial:
        """
        Record subject response to a trial.

        Returns:
            Updated trial with response
        """
        trial.subject_response_affect = response_affect
        trial.subject_response_token = response_token
        trial.response_latency_ms = response_latency_ms
        trial.responded = True

        # Score appropriateness
        score = self.scorer.score(
            trial.stimulus_affect,
            trial.stimulus_token,
            response_affect,
            response_token,
            self.syntax_graph,
        )
        trial.response_appropriate = score > 0.5  # Threshold

        self.trials.append(trial)

        return trial

    def compute_subject_metrics(
        self,
        subject_id: int,
    ) -> SubjectMetrics:
        """
        Compute metrics for a single subject.

        Returns:
            SubjectMetrics with RAS, convergence, latencies
        """
        # Filter trials for this subject
        # (In real deployment, would be tagged by subject)
        subject_trials = self.trials  # Simplified

        # Initialize metrics storage
        ras_scores = {cond: [] for cond in TestCondition}
        convergences = {cond: [] for cond in TestCondition}
        latencies = {cond: [] for cond in TestCondition}

        for trial in subject_trials:
            if not trial.responded:
                continue

            # Compute RAS if response recorded
            if trial.subject_response_affect is not None:
                ras = self.scorer.score(
                    trial.stimulus_affect,
                    trial.stimulus_token,
                    trial.subject_response_affect,
                    trial.subject_response_token,
                    self.syntax_graph,
                )
                ras_scores[trial.condition].append(ras)

            # Compute acoustic convergence if features available
            # (Would be provided in real deployment)
            # For now, skip

            # Record latency
            if trial.response_latency_ms is not None:
                latencies[trial.condition].append(trial.response_latency_ms)

        # Average scores per condition
        avg_ras = {}
        avg_convergence = {}
        for cond in TestCondition:
            avg_ras[cond] = np.mean(ras_scores[cond]) if ras_scores[cond] else 0.0
            avg_convergence[cond] = np.mean(convergences[cond]) if convergences[cond] else 0.0

        metrics = SubjectMetrics(
            ras_scores=avg_ras,
            acoustic_convergence=avg_convergence,
            response_latencies=latencies,
        )

        self.subject_metrics[subject_id] = metrics

        return metrics

    def analyze_results(self) -> Dict[str, Any]:
        """
        Perform statistical analysis of results.

        Tests:
        - RAS(A) > RAS(B) - Syntactic congruence matters
        - RAS(A) > RAS(C) - Affective congruence matters
        - RAS(B) ≈ RAS(C) - Both streams matter

        Returns:
            Analysis results with p-values
        """
        # Collect RAS scores per condition
        ras_by_condition = {cond: [] for cond in TestCondition}

        for metrics in self.subject_metrics.values():
            for cond in TestCondition:
                ras_by_condition[cond].append(metrics.ras_scores[cond])

        # Statistical tests
        # A vs B: Syntactic effect
        _, p_ab = stats.ttest_ind(
            ras_by_condition[TestCondition.CONGRUENT],
            ras_by_condition[TestCondition.SYNTACTIC_MISMATCH],
        )

        # A vs C: Affective effect
        _, p_ac = stats.ttest_ind(
            ras_by_condition[TestCondition.CONGRUENT],
            ras_by_condition[TestCondition.AFFECTIVE_MISMATCH],
        )

        # B vs C: Should not be significantly different
        _, p_bc = stats.ttest_ind(
            ras_by_condition[TestCondition.SYNTACTIC_MISMATCH],
            ras_by_condition[TestCondition.AFFECTIVE_MISMATCH],
        )

        results = {
            "ras_by_condition": {
                cond.name: np.mean(scores) for cond, scores in ras_by_condition.items()
            },
            "statistical_tests": {
                "A_vs_B_p_value": float(p_ab),
                "A_vs_C_p_value": float(p_ac),
                "B_vs_C_p_value": float(p_bc),
            },
            "significance_threshold": 0.05,
            "syntactic_effect_significant": p_ab < 0.05,
            "affective_effect_significant": p_ac < 0.05,
            "hypothesis_supported": (p_ab < 0.05) and (p_ac < 0.05) and (p_bc >= 0.05),
        }

        logger.info(f"Statistical Analysis Results: {results}")

        return results

    def save_results(self, path: str) -> None:
        """Save validation results to file."""
        import json

        results = self.analyze_results()

        # Helper to convert numpy types to Python native types
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj

        # Convert to JSON-serializable format
        output = {
            "num_subjects": self.num_subjects,
            "num_trials_per_condition": self.num_trials_per_condition,
            "results": convert_numpy(results),
            "summary": {
                "condition_A_ras": float(results["ras_by_condition"]["CONGRUENT"]),
                "condition_B_ras": float(results["ras_by_condition"]["SYNTACTIC_MISMATCH"]),
                "condition_C_ras": float(results["ras_by_condition"]["AFFECTIVE_MISMATCH"]),
                "syntactic_effect_significant": bool(results["syntactic_effect_significant"]),
                "affective_effect_significant": bool(results["affective_effect_significant"]),
                "dual_stream_hypothesis_supported": bool(results["hypothesis_supported"]),
            }
        }

        Path(path).write_text(json.dumps(output, indent=2))
        logger.info(f"Saved results to {path}")


def run_ethological_validation(
    syntax_graph: Any,
    base_affects: List[np.ndarray],
    base_tokens: List[int],
    num_subjects: int = 10,
    num_trials_per_condition: int = 20,
    output_path: str = "analysis/ethological_validation_results.json",
) -> Dict[str, Any]:
    """
    Run complete ethological validation study.

    Returns:
        Analysis results
    """
    validator = EthologicalValidator(
        syntax_graph=syntax_graph,
        num_subjects=num_subjects,
        num_trials_per_condition=num_trials_per_condition,
    )

    # Generate trial design
    trials = validator.generate_trial_design(base_affects, base_tokens)

    # Simulate responses (in real deployment, would come from actual subjects)
    np.random.seed(42)
    for trial in trials:
        # Simulate response based on condition
        if trial.condition == TestCondition.CONGRUENT:
            # High probability of appropriate response
            response_appropriate = np.random.random() > 0.3
            if response_appropriate:
                # Match affect
                response_affect = trial.stimulus_affect + np.random.randn(16) * 0.1
            else:
                # Mismatch affect
                response_affect = trial.stimulus_affect.copy()
                response_affect[0] = 1.0 - response_affect[0]

        elif trial.condition == TestCondition.SYNTACTIC_MISMATCH:
            # Low probability of syntactically appropriate response
            response_appropriate = np.random.random() > 0.7
            response_affect = trial.stimulus_affect + np.random.randn(16) * 0.1

        else:  # AFFECTIVE_MISMATCH
            # Low probability of affectively appropriate response
            response_appropriate = np.random.random() > 0.7
            if response_appropriate:
                response_affect = trial.stimulus_affect + np.random.randn(16) * 0.1
            else:
                response_affect = trial.stimulus_affect.copy()
                response_affect[0] = trial.stimulus_affect[0]  # Mismatch arousal

        # Random token from valid next
        valid_next = syntax_graph.get_valid_next_tokens(trial.stimulus_token, top_k=5)
        response_token = valid_next[np.random.randint(5)][0]

        # Record response
        validator.record_response(
            trial,
            response_affect,
            response_token,
            response_latency_ms=150 + np.random.randn() * 50,
        )

    # Compute metrics for each subject
    for subject_id in range(num_subjects):
        validator.compute_subject_metrics(subject_id)

    # Analyze results
    results = validator.analyze_results()

    # Save results
    validator.save_results(output_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from cognitive_intelligence.syntax_graph import SyntaxGraph

    # Create syntax graph
    syntax_graph = SyntaxGraph(num_tokens=64, alpha=0.01)

    # Train on some sequences
    corpus = [
        [0, 5, 12, 8],
        [5, 12, 8, 3],
        [12, 8, 3, 15],
    ] * 10
    syntax_graph.update_from_corpus(corpus)

    # Generate base stimuli
    np.random.seed(42)
    base_affects = [np.random.randn(16).astype(np.float32) for _ in range(10)]
    base_tokens = list(range(10))

    # Run validation
    results = run_ethological_validation(
        syntax_graph=syntax_graph,
        base_affects=base_affects,
        base_tokens=base_tokens,
        num_subjects=10,
        num_trials_per_condition=20,
    )

    print(f"\n=== Ethological Validation Results ===")
    print(f"Condition A (Congruent) RAS: {results['ras_by_condition']['CONGRUENT']:.3f}")
    print(f"Condition B (Syntactic Mismatch) RAS: {results['ras_by_condition']['SYNTACTIC_MISMATCH']:.3f}")
    print(f"Condition C (Affective Mismatch) RAS: {results['ras_by_condition']['AFFECTIVE_MISMATCH']:.3f}")
    print(f"\nStatistical Tests:")
    print(f"A vs B p-value: {results['statistical_tests']['A_vs_B_p_value']:.4f}")
    print(f"A vs C p-value: {results['statistical_tests']['A_vs_C_p_value']:.4f}")
    print(f"B vs C p-value: {results['statistical_tests']['B_vs_C_p_value']:.4f}")
    print(f"\nDual-Stream Hypothesis Supported: {results['hypothesis_supported']}")
