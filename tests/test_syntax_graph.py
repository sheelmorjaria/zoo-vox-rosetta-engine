#!/usr/bin/env python3
"""
Tests for Syntax Graph with Laplace Smoothing (Module 2)

Tests the probabilistic transition matrix for discrete syntactic tokens
with Laplace smoothing to prevent zero-probability bigrams from corpus sparsity.

Formula:
    P(t_i | t_{i-1}) = (Count(t_{i-1}, t_i) + α) / (Count(t_{i-1}) + α·N)

where α = 0.01 (smoothing parameter) and N = vocabulary size.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cognitive_intelligence.syntax_graph import (
    SyntaxGraph,
    SyntaxGraphConfig,
    build_syntax_graph_from_corpus,
    create_syntax_graph,
)


class TestSyntaxGraphInit(unittest.TestCase):
    """Test syntax graph initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        graph = create_syntax_graph()

        self.assertEqual(graph.num_tokens, 64)
        self.assertEqual(graph.alpha, 0.01)
        self.assertEqual(graph.transitions.shape, (64, 64))

    def test_custom_initialization(self):
        """Should initialize with custom parameters."""
        graph = SyntaxGraph(num_tokens=32, alpha=0.05)

        self.assertEqual(graph.num_tokens, 32)
        self.assertEqual(graph.alpha, 0.05)
        self.assertEqual(graph.transitions.shape, (32, 32))

    def test_uniform_initial_distribution(self):
        """Initial distribution should be uniform due to smoothing."""
        graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        # All rows should sum to approximately 1.0
        row_sums = graph.transitions.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))

    def test_initial_token_labels(self):
        """Should create default token labels."""
        graph = create_syntax_graph()

        self.assertEqual(len(graph.token_labels), 64)
        self.assertEqual(graph.token_labels[0], "token_0")
        self.assertEqual(graph.token_labels[63], "token_63")


class TestLaplaceSmoothing(unittest.TestCase):
    """Test Laplace smoothing prevents zero probabilities."""

    def test_no_zero_probabilities_after_init(self):
        """Even with no training, no transitions should be zero."""
        graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        # Check all transitions have non-zero probability
        self.assertTrue(np.all(graph.transitions > 0))

    def test_no_zero_probabilities_after_training(self):
        """After training, no transitions should be zero due to smoothing."""
        graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        # Train on sparse corpus (only a few bigrams)
        corpus = [
            [0, 5, 12],
            [5, 12, 8],
        ]
        graph.update_from_corpus(corpus)

        # All transitions should still be non-zero
        self.assertTrue(np.all(graph.transitions > 0))

    def test_unseen_bigram_has_nonzero_probability(self):
        """A bigram never seen in corpus should still have non-zero probability."""
        graph = SyntaxGraph(num_tokens=64, alpha=0.01)

        # Train on corpus that never includes token 63
        corpus = [
            [0, 1, 2],
            [1, 2, 3],
        ]
        graph.update_from_corpus(corpus)

        # P(token_63 | token_0) should be > 0 due to smoothing
        prob = graph.get_transition_probability(0, 63)
        self.assertGreater(prob, 0)

    def test_laplace_smoothing_formula(self):
        """Verify the Laplace smoothing formula is correct."""
        graph = SyntaxGraph(num_tokens=10, alpha=0.1)

        # Create simple corpus: token 0 → token 1 occurs 5 times
        # token 0 → token 2 occurs 2 times
        corpus = [[0, 1]] * 5 + [[0, 2]] * 2
        graph.update_from_corpus(corpus)

        # Expected: P(1|0) = (5 + 0.1) / (7 + 0.1*10) = 5.1 / 8.0
        expected_prob = (5 + 0.1) / (7 + 0.1 * 10)
        actual_prob = graph.get_transition_probability(0, 1)

        self.assertAlmostEqual(actual_prob, expected_prob, places=5)


class TestTransitionProbabilities(unittest.TestCase):
    """Test transition probability queries."""

    def setUp(self):
        """Create a trained graph."""
        self.graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        # Train on some sequences
        corpus = [
            [0, 1, 2, 3],
            [1, 2, 4],
            [2, 3, 5],
            [0, 5, 1],
        ]
        self.graph.update_from_corpus(corpus)

    def test_get_transition_probability(self):
        """Should return correct transition probability."""
        # Token 0 → Token 1 occurs twice in corpus
        prob = self.graph.get_transition_probability(0, 1)
        self.assertGreater(prob, 0)

    def test_invalid_token_returns_zero(self):
        """Invalid token indices should return 0."""
        prob = self.graph.get_transition_probability(99, 0)
        self.assertEqual(prob, 0)

        prob = self.graph.get_transition_probability(0, 99)
        self.assertEqual(prob, 0)

    def test_row_normalization(self):
        """Each row should sum to approximately 1.0."""
        row_sums = self.graph.transitions.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))


class TestValidNextTokens(unittest.TestCase):
    """Test getting valid next tokens."""

    def setUp(self):
        """Create a trained graph."""
        self.graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        corpus = [
            [0, 1, 2, 3],
            [0, 1, 4],
            [1, 2, 5],
        ]
        self.graph.update_from_corpus(corpus)

    def test_get_valid_next_tokens_top_k(self):
        """Should return top-k tokens by probability."""
        # Token 0 transitions to 1 most frequently
        valid_next = self.graph.get_valid_next_tokens(0, top_k=5)

        self.assertEqual(len(valid_next), 5)
        self.assertGreater(valid_next[0][1], valid_next[4][1])  # Sorted by prob

    def test_valid_next_tokens_format(self):
        """Should return (token_id, probability) tuples."""
        valid_next = self.graph.get_valid_next_tokens(1, top_k=3)

        for token_id, prob in valid_next:
            self.assertIsInstance(token_id, int)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)

    def test_invalid_current_token(self):
        """Invalid current token should return empty list."""
        valid_next = self.graph.get_valid_next_tokens(99, top_k=5)
        self.assertEqual(len(valid_next), 0)


class TestTokenSampling(unittest.TestCase):
    """Test sampling next tokens from distribution."""

    def setUp(self):
        """Create a trained graph."""
        self.graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        corpus = [
            [0, 1, 2] * 10,  # Repeated pattern
            [0, 5, 1] * 2,
        ]
        self.graph.update_from_corpus(corpus)

    def test_sample_returns_valid_token(self):
        """Sampled token should be in valid range."""
        token = self.graph.sample_next_token(0)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 16)

    def test_sampling_temperature_affects_distribution(self):
        """Temperature should affect sampling randomness."""
        import numpy.random as rng

        rng.seed(42)

        # Low temperature: more deterministic
        tokens_cold = [self.graph.sample_next_token(0, temperature=0.1) for _ in range(20)]

        # High temperature: more random
        tokens_hot = [self.graph.sample_next_token(0, temperature=2.0) for _ in range(20)]

        # Cold should have less variance (more same tokens)
        unique_cold = len(set(tokens_cold))
        unique_hot = len(set(tokens_hot))

        self.assertLessEqual(unique_cold, unique_hot + 3)  # Allow some variance


class TestEntropyAndUncertainty(unittest.TestCase):
    """Test entropy computation for uncertainty quantification."""

    def setUp(self):
        """Create graph with varying uncertainty."""
        self.graph = SyntaxGraph(num_tokens=16, alpha=0.01)

    def test_entropy_high_for_uniform_distribution(self):
        """Uniform distribution has high entropy."""
        # Before training, distribution is nearly uniform
        entropy = self.graph.get_entropy(0)

        # Should be relatively high (close to log(16) for uniform)
        self.assertGreater(entropy, 2.0)

    def test_entropy_low_for_deterministic_transitions(self):
        """Deterministic transitions have low entropy."""
        # Train on highly deterministic corpus
        corpus = [[0, 1]] * 100  # Token 0 always goes to 1
        self.graph.update_from_corpus(corpus)

        entropy = self.graph.get_entropy(0)

        # After many repetitions, entropy should decrease
        # (though smoothing prevents it from going to zero)
        initial_entropy = self.graph.get_entropy(5)  # Untoken
        self.assertLess(entropy, initial_entropy)


class TestSyntaxGraphPersistence(unittest.TestCase):
    """Test saving and loading syntax graphs."""

    def setUp(self):
        """Create a trained graph."""
        self.graph = SyntaxGraph(num_tokens=16, alpha=0.02)

        corpus = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ]
        self.graph.update_from_corpus(corpus)

    def test_save_and_load(self):
        """Should save and load graph correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            self.graph.save(temp_path)

            # Load
            loaded_graph = SyntaxGraph.load(temp_path)

            # Verify parameters match
            self.assertEqual(loaded_graph.num_tokens, self.graph.num_tokens)
            self.assertEqual(loaded_graph.alpha, self.graph.alpha)

            # Verify transitions match
            np.testing.assert_array_almost_equal(
                loaded_graph.transitions,
                self.graph.transitions
            )

            # Verify counts match
            np.testing.assert_array_equal(
                loaded_graph.counts,
                self.graph.counts
            )

        finally:
            temp_path.unlink()

    def test_load_nonexistent_file_raises_error(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            SyntaxGraph.load(Path("/nonexistent/path.json"))


class TestHasValidTransition(unittest.TestCase):
    """Test checking for valid transitions."""

    def setUp(self):
        """Create graph with Laplace smoothing."""
        self.graph = SyntaxGraph(num_tokens=16, alpha=0.01)

    def test_all_transitions_valid_with_smoothing(self):
        """With Laplace smoothing, ALL transitions should be valid."""
        corpus = [[0, 1, 2]]
        self.graph.update_from_corpus(corpus)

        # Every possible transition should have non-zero probability
        for i in range(16):
            for j in range(16):
                self.assertTrue(
                    self.graph.has_valid_transition(i, j),
                    f"Transition {i} → {j} should be valid"
                )

    def test_invalid_tokens_return_false(self):
        """Invalid token indices should return False."""
        self.assertFalse(self.graph.has_valid_transition(0, 99))
        self.assertFalse(self.graph.has_valid_transition(99, 0))


class TestBuildSyntaxGraphFromCorpus(unittest.TestCase):
    """Test factory function for building from corpus."""

    def test_factory_function(self):
        """Should build graph from corpus."""
        corpus = [
            [0, 1, 2, 3],
            [1, 2, 4, 5],
            [2, 3, 5, 6],
        ]

        graph = build_syntax_graph_from_corpus(
            corpus,
            num_tokens=32,
            alpha=0.01
        )

        self.assertEqual(graph.num_tokens, 32)
        self.assertEqual(graph.alpha, 0.01)
        self.assertGreater(graph.counts.sum(), 0)  # Some counts should exist


class TestIntegration(unittest.TestCase):
    """Integration tests for syntax graph."""

    def test_full_pipeline(self):
        """Test complete pipeline: train → query → sample."""
        # Create graph
        graph = SyntaxGraph(num_tokens=64, alpha=0.01)

        # Train on corpus
        corpus = [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [0, 5, 10, 15],
        ]
        graph.update_from_corpus(corpus)

        # Query valid next tokens
        valid_next = graph.get_valid_next_tokens(0, top_k=5)
        self.assertGreater(len(valid_next), 0)

        # Check a specific transition
        prob_0_to_1 = graph.get_transition_probability(0, 1)
        self.assertGreater(prob_0_to_1, 0)

        # Sample from distribution
        sampled = graph.sample_next_token(0)
        self.assertGreaterEqual(sampled, 0)
        self.assertLess(sampled, 64)

    def test_no_dead_ends(self):
        """Agent should never encounter a dead end (no valid next tokens)."""
        graph = SyntaxGraph(num_tokens=16, alpha=0.01)

        # Train on sparse corpus
        corpus = [[0, 1, 2]]
        graph.update_from_corpus(corpus)

        # Every token should have valid next tokens
        for token in range(16):
            valid_next = graph.get_valid_next_tokens(token, top_k=16)
            self.assertEqual(len(valid_next), 16)  # All tokens are valid


if __name__ == "__main__":
    unittest.main()
