#!/usr/bin/env python3
"""
Tests for Syntax Graph Builder

Tests the syntax graph building pipeline including:
- Corpus tokenization
- Transition matrix building
- Laplace smoothing
- Syntax graph saving/loading

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cognitive_intelligence.build_syntax_graph import (
    SyntaxGraphBuilderConfig,
    SyntaxGraphBuilder,
    CorpusTokenizer,
)
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE, VQVAECheckpoint
from cognitive_intelligence.syntax_graph import SyntaxGraph


class TestSyntaxGraphBuilderConfig(unittest.TestCase):
    """Test builder configuration."""

    def test_default_config(self):
        """Should create default config."""
        config = SyntaxGraphBuilderConfig()

        self.assertEqual(config.alpha, 0.01)
        self.assertEqual(config.num_tokens, 64)

    def test_custom_config(self):
        """Should accept custom parameters."""
        config = SyntaxGraphBuilderConfig(
            alpha=0.05,
            token_labels=["contact", "alarm", "territorial"],
        )

        self.assertEqual(config.alpha, 0.05)
        self.assertEqual(len(config.token_labels), 3)


class TestCorpusTokenizer(unittest.TestCase):
    """Test corpus tokenization."""

    def setUp(self):
        """Create VQ-VAE model for tokenization."""
        self.device = torch.device("cpu")
        self.vqvae = SyntacticVQVAE(
            input_dim=44,
            codebook_size=8,  # Small for testing
            codebook_dim=16,
            hidden_dim=32,
        )
        self.vqvae.eval()
        self.tokenizer = CorpusTokenizer(self.vqvae, self.device)

    def test_tokenize_features_112d(self):
        """Should tokenize single feature vector."""
        features = np.random.randn(112).astype(np.float32)

        token_id = self.tokenizer.tokenize_features_112d(features)

        self.assertIsInstance(token_id, int)
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, 8)

    def test_tokenize_batch(self):
        """Should tokenize batch of features."""
        features = np.random.randn(10, 112).astype(np.float32)

        tokens = self.tokenizer.tokenize_batch(features)

        self.assertEqual(len(tokens), 10)
        for token in tokens:
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, 8)

    def test_tokenize_segments_json(self):
        """Should tokenize segments from JSON."""
        # Create temporary JSON file
        temp_json = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

        segments = {
            "segments": [
                {
                    "features_112d": np.random.randn(112).tolist(),
                    "species": "marmoset",
                    "phrase_id": f"test_{i}",
                }
                for i in range(20)
            ]
        }

        json.dump(segments, temp_json)
        temp_json.close()

        try:
            sequences = self.tokenizer.tokenize_segments(temp_json.name)

            self.assertEqual(len(sequences), 20)
            for seq in sequences:
                self.assertEqual(len(seq), 1)  # Single token per segment
        finally:
            Path(temp_json.name).unlink(missing_ok=True)


class TestSyntaxGraphBuilder(unittest.TestCase):
    """Test syntax graph building."""

    def setUp(self):
        """Create builder config."""
        self.config = SyntaxGraphBuilderConfig(alpha=0.01)
        self.builder = SyntaxGraphBuilder(self.config)

    def test_build_from_sequences(self):
        """Should build graph from token sequences."""
        # Create mock sequences
        sequences = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 0],
            [0, 1, 2, 3],
        ]

        graph = self.builder.build_from_sequences(sequences, num_tokens=4)

        self.assertIsNotNone(graph)
        self.assertEqual(graph.num_tokens, 4)

    def test_laplace_smoothing(self):
        """Should apply Laplace smoothing (no zero probabilities)."""
        sequences = [[0, 1], [1, 2]]

        graph = self.builder.build_from_sequences(sequences, num_tokens=4)

        # Check no zero probabilities
        for i in range(4):
            probs = graph.transitions[i]
            for prob in probs:
                self.assertGreater(prob, 0)

    def test_get_statistics(self):
        """Should compute correct statistics."""
        sequences = [
            [0, 1, 2],
            [1, 2, 3],
            [0, 1],
        ]

        graph = self.builder.build_from_sequences(sequences, num_tokens=4)
        stats = graph.get_statistics()

        self.assertIn("num_tokens", stats)
        self.assertIn("min_probability", stats)
        self.assertIn("max_probability", stats)
        self.assertGreater(stats["num_tokens"], 0)

    def test_build_and_save(self):
        """Should build and save syntax graph."""
        sequences = [[0, 1], [1, 2], [2, 0]]

        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / "test_syntax_graph.json"

        try:
            graph = self.builder.build_and_save(
                sequences, num_tokens=3, output_path=str(output_path)
            )

            self.assertTrue(output_path.exists())

            # Load and verify
            loaded_graph = SyntaxGraph.load_json(str(output_path))
            self.assertEqual(loaded_graph.num_tokens, 3)
        finally:
            # Clean up
            if output_path.exists():
                output_path.unlink()
            Path(temp_dir).rmdir()


class TestSyntaxGraphUsage(unittest.TestCase):
    """Test syntax graph usage for validation."""

    def setUp(self):
        """Create test syntax graph."""
        # Create a simple graph with known transitions
        transitions = np.array([
            [0.1, 0.7, 0.1, 0.1],  # 0 -> 1 is most likely
            [0.1, 0.1, 0.7, 0.1],  # 1 -> 2 is most likely
            [0.1, 0.1, 0.1, 0.7],  # 2 -> 3 is most likely
            [0.7, 0.1, 0.1, 0.1],  # 3 -> 0 is most likely
        ])

        self.graph = SyntaxGraph(num_tokens=4)
        self.graph.transitions = transitions

    def test_get_valid_next_tokens(self):
        """Should return valid next tokens sorted by probability."""
        next_tokens = self.graph.get_valid_next_tokens(0, top_k=2)

        self.assertEqual(len(next_tokens), 2)
        self.assertEqual(next_tokens[0][0], 1)  # Most likely
        self.assertGreater(next_tokens[0][1], next_tokens[1][1])

    def test_is_valid_bigram(self):
        """Should validate bigrams."""
        # Valid bigram (high probability)
        self.assertTrue(self.graph.is_valid_bigram(0, 1, threshold=0.5))

        # Invalid bigram (low probability)
        self.assertFalse(self.graph.is_valid_bigram(0, 0, threshold=0.5))

    def test_compute_sequence_probability(self):
        """Should compute log sequence probability."""
        sequence = [0, 1, 2, 3]

        log_prob = self.graph.compute_sequence_probability(sequence)

        # Log probability should be finite (can be negative)
        self.assertGreater(log_prob, -10)  # Not extremely unlikely

    def test_sample_next_token(self):
        """Should sample next token from distribution."""
        token = self.graph.sample_next_token(0)

        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 4)

    def test_generate_sequence(self):
        """Should generate sequence from start token."""
        sequence = self.graph.generate_sequence(start_token=0, max_length=10)

        self.assertGreater(len(sequence), 0)
        self.assertLessEqual(len(sequence), 10)
        self.assertEqual(sequence[0], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""

    def test_end_to_end_tokenization_and_graph_building(self):
        """Should tokenize corpus and build syntax graph."""
        # Create VQ-VAE
        device = torch.device("cpu")
        vqvae = SyntacticVQVAE(
            input_dim=44,
            codebook_size=8,
            codebook_dim=16,
            hidden_dim=32,
        )
        vqvae.eval()

        # Create tokenizer
        tokenizer = CorpusTokenizer(vqvae, device)

        # Create mock segments
        temp_json = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

        segments = {
            "segments": [
                {
                    "features_112d": np.random.randn(112).tolist(),
                    "species": "bat",
                    "phrase_id": f"phrase_{i}",
                }
                for i in range(30)
            ]
        }

        json.dump(segments, temp_json)
        temp_json.close()

        try:
            # Tokenize
            sequences = tokenizer.tokenize_segments(temp_json.name)

            # Build graph
            builder = SyntaxGraphBuilder(SyntaxGraphBuilderConfig(alpha=0.01))
            graph = builder.build_from_sequences(sequences, num_tokens=8)

            # Verify
            self.assertEqual(graph.num_tokens, 8)
            stats = graph.get_statistics()
            self.assertGreater(stats["num_tokens"], 0)
        finally:
            Path(temp_json.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
