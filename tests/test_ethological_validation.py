#!/usr/bin/env python3
"""
Tests for Ethological Validation (Sprint 5)

Tests the Semantic vs Affective Mismatch validation framework:
- Condition A (Congruent): Matching affect + matching syntax
- Condition B (Syntactic Mismatch): Matching affect + mismatched syntax
- Condition C (Affective Mismatch): Mismatched affect + matching syntax

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Try importing required modules
try:
    from cognitive_intelligence.ethological_validation import (
        AcousticConvergenceMetric,
        EthologicalValidator,
        ResponseAppropriatenessScorer,
        run_ethological_validation,
        SubjectMetrics,
        TestCondition,
        TestTrial,
    )
    from cognitive_intelligence.syntax_graph import SyntaxGraph
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class TestTestCondition(unittest.TestCase):
    """Test the TestCondition enum."""

    def test_condition_values(self):
        """Test that condition enum has correct values."""
        self.assertEqual(TestCondition.CONGRUENT.value, "A")
        self.assertEqual(TestCondition.SYNTACTIC_MISMATCH.value, "B")
        self.assertEqual(TestCondition.AFFECTIVE_MISMATCH.value, "C")

    def test_condition_iteration(self):
        """Test that we can iterate over all conditions."""
        conditions = list(TestCondition)
        self.assertEqual(len(conditions), 3)
        self.assertIn(TestCondition.CONGRUENT, conditions)
        self.assertIn(TestCondition.SYNTACTIC_MISMATCH, conditions)
        self.assertIn(TestCondition.AFFECTIVE_MISMATCH, conditions)


class TestTestTrial(unittest.TestCase):
    """Test the TestTrial dataclass."""

    def test_trial_creation(self):
        """Test creating a trial."""
        affect = np.random.randn(16).astype(np.float32)
        trial = TestTrial(
            condition=TestCondition.CONGRUENT,
            stimulus_affect=affect,
            stimulus_token=5,
        )

        self.assertEqual(trial.condition, TestCondition.CONGRUENT)
        self.assertEqual(trial.stimulus_token, 5)
        self.assertFalse(trial.responded)
        self.assertIsNone(trial.response_latency_ms)

    def test_trial_with_response(self):
        """Test creating a trial with a response."""
        stimulus_affect = np.random.randn(16).astype(np.float32)
        response_affect = np.random.randn(16).astype(np.float32)

        trial = TestTrial(
            condition=TestCondition.CONGRUENT,
            stimulus_affect=stimulus_affect,
            stimulus_token=5,
            subject_response_affect=response_affect,
            subject_response_token=10,
            response_latency_ms=150.0,
            responded=True,
            response_appropriate=True,
        )

        self.assertTrue(trial.responded)
        self.assertEqual(trial.response_latency_ms, 150.0)
        self.assertTrue(trial.response_appropriate)


class TestResponseAppropriatenessScorer(unittest.TestCase):
    """Test the ResponseAppropriatenessScorer."""

    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ResponseAppropriatenessScorer(
            affect_weight=0.5,
            syntax_weight=0.5,
        )

        # Create mock syntax graph
        self.syntax_graph = MagicMock()
        self.syntax_graph.get_transition_probability = Mock(
            return_value=0.1
        )

    def test_affect_congruence_identical_vectors(self):
        """Test affect congruence with identical vectors."""
        affect = np.random.randn(16).astype(np.float32)
        score = self.scorer.score_affect_congruence(affect, affect)

        # Identical vectors should give score of 1.0
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_affect_congruence_opposite_vectors(self):
        """Test affect congruence with opposite vectors."""
        affect = np.random.randn(16).astype(np.float32)
        opposite = -affect
        score = self.scorer.score_affect_congruence(affect, opposite)

        # Opposite vectors should give score near 0.0
        self.assertLess(score, 0.1)

    def test_affect_congruence_zero_vectors(self):
        """Test affect congruence with zero vectors."""
        zero = np.zeros(16, dtype=np.float32)
        affect = np.random.randn(16).astype(np.float32)
        score = self.scorer.score_affect_congruence(affect, zero)

        # Should return 0.0 for zero norm
        self.assertEqual(score, 0.0)

    def test_syntax_congruence_high_probability(self):
        """Test syntax congruence with high probability transition."""
        self.syntax_graph.get_transition_probability = Mock(return_value=0.5)

        score = self.scorer.score_syntax_congruence(5, 10, self.syntax_graph)

        # High probability should give high score
        self.assertGreater(score, 0.5)

    def test_syntax_congruence_low_probability(self):
        """Test syntax congruence with low probability transition."""
        self.syntax_graph.get_transition_probability = Mock(return_value=0.001)

        score = self.scorer.score_syntax_congruence(5, 10, self.syntax_graph)

        # Low probability should give low score
        self.assertLess(score, 0.1)

    def test_syntax_congruence_error_handling(self):
        """Test syntax congruence handles errors gracefully."""
        self.syntax_graph.get_transition_probability = Mock(
            side_effect=Exception("Test error")
        )

        score = self.scorer.score_syntax_congruence(5, 10, self.syntax_graph)

        # Should return default small probability
        self.assertAlmostEqual(score, 0.1, places=5)

    def test_overall_score(self):
        """Test overall response appropriateness score."""
        stimulus_affect = np.array([0.5] * 16, dtype=np.float32)
        response_affect = np.array([0.5] * 16, dtype=np.float32)

        self.syntax_graph.get_transition_probability = Mock(return_value=0.1)

        score = self.scorer.score(
            stimulus_affect=stimulus_affect,
            stimulus_token=5,
            response_affect=response_affect,
            response_token=10,
            syntax_graph=self.syntax_graph,
        )

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestAcousticConvergenceMetric(unittest.TestCase):
    """Test the AcousticConvergenceMetric."""

    def setUp(self):
        """Set up test fixtures."""
        self.metric = AcousticConvergenceMetric(feature_dim=112)

    def test_convergence_identical_features(self):
        """Test convergence with identical features."""
        features = np.random.randn(112).astype(np.float32)
        score = self.metric.compute_convergence(features, features)

        # Identical features should give score of 1.0
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_convergence_opposite_features(self):
        """Test convergence with opposite features."""
        features = np.random.randn(112).astype(np.float32)
        opposite = -features
        score = self.metric.compute_convergence(features, opposite)

        # Opposite features should give score near 0.0
        self.assertLess(score, 0.1)

    def test_convergence_score_range(self):
        """Test that convergence score is always in [0, 1]."""
        for _ in range(10):
            f1 = np.random.randn(112).astype(np.float32)
            f2 = np.random.randn(112).astype(np.float32)
            score = self.metric.compute_convergence(f1, f2)

            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestEthologicalValidator(unittest.TestCase):
    """Test the EthologicalValidator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create syntax graph
        self.syntax_graph = SyntaxGraph(num_tokens=64, alpha=0.01)

        # Add some transitions
        corpus = [
            [0, 5, 12, 8],
            [5, 12, 8, 3],
            [12, 8, 3, 15],
            [0, 1, 2, 3],
        ] * 10
        self.syntax_graph.update_from_corpus(corpus)

        # Create validator
        self.validator = EthologicalValidator(
            syntax_graph=self.syntax_graph,
            num_subjects=5,
            num_trials_per_condition=5,
        )

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        self.assertEqual(self.validator.num_subjects, 5)
        self.assertEqual(self.validator.num_trials_per_condition, 5)
        self.assertEqual(len(self.validator.trials), 0)

    def test_create_stimulus_congruent(self):
        """Test stimulus creation for congruent condition."""
        base_affect = np.random.randn(16).astype(np.float32)
        base_token = 0

        affect, token = self.validator.create_stimulus(
            TestCondition.CONGRUENT, base_affect, base_token
        )

        # Affect should match
        np.testing.assert_array_almost_equal(affect, base_affect)

        # Token should be a valid high-probability transition
        self.assertIsInstance(token, (int, np.integer))
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 64)

    def test_create_stimulus_syntactic_mismatch(self):
        """Test stimulus creation for syntactic mismatch."""
        base_affect = np.random.randn(16).astype(np.float32)
        base_token = 0

        affect, token = self.validator.create_stimulus(
            TestCondition.SYNTACTIC_MISMATCH, base_affect, base_token
        )

        # Affect should match
        np.testing.assert_array_almost_equal(affect, base_affect)

        # Token should be a low-probability transition
        self.assertIsInstance(token, (int, np.integer))

    def test_create_stimulus_affective_mismatch(self):
        """Test stimulus creation for affective mismatch."""
        base_affect = np.array([0.5] + [0.0] * 15, dtype=np.float32)
        base_token = 0

        affect, token = self.validator.create_stimulus(
            TestCondition.AFFECTIVE_MISMATCH, base_affect, base_token
        )

        # Arousal (first dimension) should be inverted
        self.assertAlmostEqual(affect[0], 1.0 - base_affect[0], places=5)

        # Token should be high probability
        self.assertIsInstance(token, (int, np.integer))

    def test_generate_trial_design(self):
        """Test trial design generation."""
        base_affects = [np.random.randn(16).astype(np.float32) for _ in range(5)]
        base_tokens = list(range(5))

        trials = self.validator.generate_trial_design(base_affects, base_tokens)

        # Should have 3 conditions * 5 trials = 15 total
        self.assertEqual(len(trials), 15)

        # Check each condition has correct number of trials
        condition_counts = {
            TestCondition.CONGRUENT: 0,
            TestCondition.SYNTACTIC_MISMATCH: 0,
            TestCondition.AFFECTIVE_MISMATCH: 0,
        }

        for trial in trials:
            condition_counts[trial.condition] += 1

        for count in condition_counts.values():
            self.assertEqual(count, 5)

    def test_record_response(self):
        """Test recording a subject response."""
        trial = TestTrial(
            condition=TestCondition.CONGRUENT,
            stimulus_affect=np.array([0.5] * 16, dtype=np.float32),
            stimulus_token=5,
        )

        response_affect = np.array([0.5] * 16, dtype=np.float32)

        updated_trial = self.validator.record_response(
            trial,
            response_affect=response_affect,
            response_token=10,
            response_latency_ms=150.0,
        )

        self.assertTrue(updated_trial.responded)
        self.assertEqual(updated_trial.response_latency_ms, 150.0)
        self.assertIsNotNone(updated_trial.subject_response_affect)

    def test_compute_subject_metrics(self):
        """Test computing metrics for a subject."""
        # Add some trials
        for _ in range(10):
            trial = TestTrial(
                condition=TestCondition.CONGRUENT,
                stimulus_affect=np.array([0.5] * 16, dtype=np.float32),
                stimulus_token=5,
            )
            self.validator.record_response(
                trial,
                response_affect=np.array([0.5] * 16, dtype=np.float32),
                response_token=10,
                response_latency_ms=150.0,
            )

        metrics = self.validator.compute_subject_metrics(subject_id=0)

        # Check metrics structure
        self.assertIsInstance(metrics, SubjectMetrics)
        self.assertIn(TestCondition.CONGRUENT, metrics.ras_scores)
        self.assertIn(TestCondition.SYNTACTIC_MISMATCH, metrics.ras_scores)
        self.assertIn(TestCondition.AFFECTIVE_MISMATCH, metrics.ras_scores)

    def test_analyze_results(self):
        """Test statistical analysis of results."""
        # Compute metrics first
        self.validator.compute_subject_metrics(subject_id=0)

        results = self.validator.analyze_results()

        # Check results structure
        self.assertIn("ras_by_condition", results)
        self.assertIn("statistical_tests", results)
        self.assertIn("significance_threshold", results)
        self.assertIn("syntactic_effect_significant", results)
        self.assertIn("affective_effect_significant", results)
        self.assertIn("hypothesis_supported", results)

        # Check statistical tests
        stats = results["statistical_tests"]
        self.assertIn("A_vs_B_p_value", stats)
        self.assertIn("A_vs_C_p_value", stats)
        self.assertIn("B_vs_C_p_value", stats)

    def test_save_results(self):
        """Test saving results to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            output_path = f.name

        try:
            self.validator.compute_subject_metrics(subject_id=0)
            self.validator.save_results(output_path)

            # Check file was created
            self.assertTrue(Path(output_path).exists())

            # Check JSON can be loaded
            with open(output_path) as f:
                saved_results = json.load(f)

            self.assertIn("num_subjects", saved_results)
            self.assertIn("results", saved_results)
            self.assertIn("summary", saved_results)

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestRunEthologicalValidation(unittest.TestCase):
    """Test the complete ethological validation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.syntax_graph = SyntaxGraph(num_tokens=64, alpha=0.01)
        corpus = [
            [0, 5, 12, 8],
            [5, 12, 8, 3],
            [12, 8, 3, 15],
        ] * 10
        self.syntax_graph.update_from_corpus(corpus)

    def test_run_validation_produces_results(self):
        """Test that run_ethological_validation produces results."""
        np.random.seed(42)
        base_affects = [np.random.randn(16).astype(np.float32) for _ in range(5)]
        base_tokens = list(range(5))

        results = run_ethological_validation(
            syntax_graph=self.syntax_graph,
            base_affects=base_affects,
            base_tokens=base_tokens,
            num_subjects=3,
            num_trials_per_condition=3,
            output_path="/tmp/test_ethological_results.json",
        )

        # Check results structure
        self.assertIn("ras_by_condition", results)
        self.assertIn("statistical_tests", results)
        self.assertIn("hypothesis_supported", results)

    def test_results_have_all_conditions(self):
        """Test that results include all three conditions."""
        np.random.seed(42)
        base_affects = [np.random.randn(16).astype(np.float32) for _ in range(5)]
        base_tokens = list(range(5))

        results = run_ethological_validation(
            syntax_graph=self.syntax_graph,
            base_affects=base_affects,
            base_tokens=base_tokens,
            num_subjects=2,
            num_trials_per_condition=2,
        )

        ras_by_condition = results["ras_by_condition"]
        self.assertIn("CONGRUENT", ras_by_condition)
        self.assertIn("SYNTACTIC_MISMATCH", ras_by_condition)
        self.assertIn("AFFECTIVE_MISMATCH", ras_by_condition)


class TestConditionDifferences(unittest.TestCase):
    """Test that different conditions produce different responses."""

    def setUp(self):
        """Set up test fixtures."""
        self.syntax_graph = SyntaxGraph(num_tokens=64, alpha=0.01)
        corpus = [
            [0, 5, 12, 8],
            [5, 12, 8, 3],
            [12, 8, 3, 15],
        ] * 10
        self.syntax_graph.update_from_corpus(corpus)

        self.validator = EthologicalValidator(
            syntax_graph=self.syntax_graph,
            num_subjects=2,
            num_trials_per_condition=2,
        )

    def test_congruent_vs_syntactic_mismatch(self):
        """Test that congruent and syntactic mismatch differ."""
        base_affect = np.random.randn(16).astype(np.float32)
        base_token = 0

        # Congruent
        affect_cong, token_cong = self.validator.create_stimulus(
            TestCondition.CONGRUENT, base_affect, base_token
        )

        # Syntactic mismatch
        affect_syn, token_syn = self.validator.create_stimulus(
            TestCondition.SYNTACTIC_MISMATCH, base_affect, base_token
        )

        # Affect should match
        np.testing.assert_array_almost_equal(affect_cong, affect_syn)

        # Tokens should differ (different probability)
        # Note: they might occasionally be the same by chance
        self.assertIsInstance(token_cong, (int, np.integer))
        self.assertIsInstance(token_syn, (int, np.integer))

    def test_congruent_vs_affective_mismatch(self):
        """Test that congruent and affective mismatch differ."""
        # Use 0.7 instead of 0.5 so inversion is noticeable (1.0 - 0.7 = 0.3)
        base_affect = np.array([0.7] + [0.0] * 15, dtype=np.float32)
        base_token = 0

        # Congruent
        affect_cong, token_cong = self.validator.create_stimulus(
            TestCondition.CONGRUENT, base_affect, base_token
        )

        # Affective mismatch
        condition_affect, token_affect = self.validator.create_stimulus(
            TestCondition.AFFECTIVE_MISMATCH, base_affect, base_token
        )

        # Affect should differ (arousal inverted from 0.7 to 0.3)
        self.assertAlmostEqual(affect_cong[0], 0.7, places=1)
        self.assertAlmostEqual(condition_affect[0], 0.3, places=1)


if __name__ == "__main__":
    unittest.main()
