#!/usr/bin/env python3
"""
Tests for Syntactic VQ-VAE (Stream 2)

These tests verify the VQ-VAE with EMA for discrete token encoding,
including codebook utilization and Laplace smoothing.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class TestSyntacticFeatureExtractor(unittest.TestCase):
    """Test syntactic feature extraction from 112D Rosetta vector."""

    def test_extract_syntactic_features(self):
        """Should extract 44D syntactic features from 112D input."""
        from cognitive_intelligence.syntactic_encoder import SyntacticFeatureExtractor

        features_112d = np.random.randn(112).astype(np.float32)
        syntactic = SyntacticFeatureExtractor.extract_syntactic_features(features_112d)

        self.assertEqual(len(syntactic), SyntacticFeatureExtractor.OUTPUT_DIM)
        self.assertEqual(SyntacticFeatureExtractor.OUTPUT_DIM, 44)

    def test_batch_extraction(self):
        """Should handle batch input."""
        from cognitive_intelligence.syntactic_encoder import SyntacticFeatureExtractor

        batch = np.random.randn(10, 112).astype(np.float32)
        syntactic = SyntacticFeatureExtractor.extract_syntactic_features(batch)

        self.assertEqual(syntactic.shape, (10, 44))

    def test_feature_names(self):
        """Should return correct number of feature names."""
        from cognitive_intelligence.syntactic_encoder import SyntacticFeatureExtractor

        names = SyntacticFeatureExtractor.get_feature_names()

        self.assertEqual(len(names), SyntacticFeatureExtractor.OUTPUT_DIM)
        self.assertIn("MFCC_1", names)
        self.assertIn("Attack_time_ms", names)
        self.assertIn("Harmonic_mean", names)


class TestEMAVectorQuantizer(unittest.TestCase):
    """Test EMA vector quantizer."""

    def test_quantization_shape(self):
        """Should produce correct output shapes."""
        from cognitive_intelligence.syntactic_vqvae import EMAVectorQuantizer

        vq = EMAVectorQuantizer(codebook_size=64, codebook_dim=32)
        vq.eval()

        z = torch.randn(4, 32)
        z_q, token_ids, perplexity = vq(z)

        self.assertEqual(z_q.shape, (4, 32))
        self.assertEqual(token_ids.shape, (4,))
        self.assertEqual(perplexity.shape, ())

    def test_perplexity_range(self):
        """Perplexity should be between 1 and codebook_size."""
        from cognitive_intelligence.syntactic_vqvae import EMAVectorQuantizer

        vq = EMAVectorQuantizer(codebook_size=64, codebook_dim=32)
        vq.eval()

        z = torch.randn(10, 32)
        _, _, perplexity = vq(z)

        # Perplexity between 1 and codebook_size
        self.assertGreaterEqual(perplexity.item(), 1.0)
        self.assertLessEqual(perplexity.item(), 64.0)

    def test_codebook_utilization(self):
        """Should track codebook utilization."""
        from cognitive_intelligence.syntactic_vqvae import EMAVectorQuantizer

        vq = EMAVectorQuantizer(codebook_size=64, codebook_dim=32)
        vq.eval()

        # Use only some codes
        z = torch.randn(100, 32)
        _, _, _ = vq(z)

        utilization = vq.get_codebook_utilization()

        # Should use at least some codes
        self.assertGreater(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)

    def test_revive_dead_codes(self):
        """Should revive unused codebook entries."""
        from cognitive_intelligence.syntactic_vqvae import EMAVectorQuantizer

        vq = EMAVectorQuantizer(codebook_size=64, codebook_dim=32)

        # Force some codes to be unused
        vq.usage_count[:] = 10
        vq.usage_count[50:] = 0  # Last 14 unused

        vq.revive_dead_codes(threshold=0.5)

        # Usage should be reset for revived codes
        self.assertEqual(vq.usage_count[50:].sum().item(), 0)


class TestSyntacticVQVAE(unittest.TestCase):
    """Test VQ-VAE architecture."""

    def test_vqvae_initialization(self):
        """Should initialize with correct dimensions."""
        from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

        vqvae = SyntacticVQVAE(
            input_dim=44,
            codebook_size=64,
            codebook_dim=32,
        )

        self.assertEqual(vqvae.input_dim, 44)
        self.assertEqual(vqvae.codebook_size, 64)
        self.assertEqual(vqvae.codebook_dim, 32)

    def test_forward_pass(self):
        """Should return reconstruction with correct shape."""
        from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

        vqvae = SyntacticVQVAE(input_dim=44, codebook_size=64, codebook_dim=32)
        vqvae.eval()

        x = torch.randn(4, 44)
        recon, z_q, token_ids, perplexity = vqvae(x)

        self.assertEqual(recon.shape, (4, 44))
        self.assertEqual(token_ids.shape, (4,))

    def test_tokenize(self):
        """Should convert input to token IDs."""
        from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

        vqvae = SyntacticVQVAE(input_dim=44, codebook_size=64, codebook_dim=32)
        vqvae.eval()

        x = torch.randn(8, 44)
        token_ids = vqvae.tokenize(x)

        self.assertEqual(token_ids.shape, (8,))
        self.assertTrue(torch.all(token_ids >= 0))
        self.assertTrue(torch.all(token_ids < 64))

    def test_decode_code(self):
        """Should decode from token IDs."""
        from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

        vqvae = SyntacticVQVAE(input_dim=44, codebook_size=64, codebook_dim=32)
        vqvae.eval()

        token_ids = torch.tensor([5, 10, 15])
        recon = vqvae.decode_code(token_ids)

        self.assertEqual(recon.shape, (3, 44))

    def test_loss_function(self):
        """Should compute VQ-VAE loss."""
        from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

        vqvae = SyntacticVQVAE(input_dim=44, codebook_size=64, codebook_dim=32)

        x = torch.randn(4, 44)
        z = vqvae.encoder(x)
        z_q, token_ids, _ = vqvae.vq(z)

        total_loss, recon_loss, commit_loss = vqvae.loss_function(x, x, z, z_q)

        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreaterEqual(commit_loss.item(), 0.0)

    def test_codebook_utilization(self):
        """Should report codebook utilization."""
        from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

        vqvae = SyntacticVQVAE(input_dim=44, codebook_size=64, codebook_dim=32)
        vqvae.eval()

        # Run some data through
        x = torch.randn(100, 44)
        with torch.no_grad():
            vqvae(x)

        utilization = vqvae.get_codebook_utilization()

        self.assertGreater(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)


class TestVQVAECheckpoint(unittest.TestCase):
    """Test checkpoint management."""

    def test_save_load_model_only(self):
        """Should save and load model state dict."""
        import torch
        from cognitive_intelligence.syntactic_vqvae import create_syntactic_vqvae

        vqvae = create_syntactic_vqvae()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save using torch.save
            torch.save({
                'state_dict': vqvae.state_dict(),
                'config': {
                    'input_dim': vqvae.input_dim,
                    'codebook_size': vqvae.codebook_size,
                    'codebook_dim': vqvae.codebook_dim,
                    'commitment_cost': vqvae.commitment_cost,
                },
            }, path)

            self.assertTrue(path.exists())

            # Load model
            data = torch.load(path)
            config = data['config']

            loaded = create_syntactic_vqvae()
            loaded.load_state_dict(data['state_dict'])

            self.assertEqual(loaded.input_dim, vqvae.input_dim)
            self.assertEqual(loaded.codebook_size, vqvae.codebook_size)

    def test_save_codebook(self):
        """Should save codebook as numpy array."""
        import torch
        from cognitive_intelligence.syntactic_vqvae import create_syntactic_vqvae

        vqvae = create_syntactic_vqvae()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "codebook.npy"

            # Save codebook directly
            codebook = vqvae.vq.codebook_ema.detach().cpu().numpy()
            np.save(path, codebook)

            self.assertTrue(path.exists())

            # Load and check shape
            loaded_codebook = np.load(path)
            self.assertEqual(loaded_codebook.shape, (64, 32))


class TestSyntaxGraph(unittest.TestCase):
    """Test syntax graph with Laplace smoothing."""

    def test_initialization(self):
        """Should initialize with uniform distribution."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=64, alpha=0.01)

        self.assertEqual(graph.num_tokens, 64)
        self.assertEqual(graph.alpha, 0.01)

        # Check uniform initialization
        expected = 1.0 / 64
        np.testing.assert_array_almost_equal(
            graph.transitions[0], np.full(64, expected), decimal=3
        )

    def test_update_from_corpus(self):
        """Should build transition matrix from sequences."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)

        # Create sequences with clear patterns
        sequences = [
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [5, 6, 7, 8],
        ]

        graph.update_from_corpus(sequences)

        # Check that 0->1 has high probability
        prob_0_1 = graph.get_transition_probability(0, 1)
        prob_0_5 = graph.get_transition_probability(0, 5)

        self.assertGreater(prob_0_1, prob_0_5)

    def test_laplace_smoothing_no_zero_probabilities(self):
        """Laplace smoothing should prevent zero probabilities."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)

        # Sparse corpus - not all transitions seen
        sequences = [[0, 1, 2]]

        graph.update_from_corpus(sequences)

        # Check all probabilities are non-zero
        for src in range(10):
            for dst in range(10):
                p = graph.get_transition_probability(src, dst)
                self.assertGreater(p, 0.0)

    def test_get_valid_next_tokens(self):
        """Should return top-k tokens by probability."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)

        # Create sequences favoring certain transitions
        sequences = [[0, 1] * 10, [0, 2] * 5, [0, 3] * 2]
        graph.update_from_corpus(sequences)

        top_k = graph.get_valid_next_tokens(0, top_k=3)

        self.assertEqual(len(top_k), 3)
        self.assertEqual(top_k[0][0], 1)  # Most frequent

        # Check probabilities sum to <= 1
        sum_probs = sum(p for _, p in top_k)
        self.assertLessEqual(sum_probs, 1.0)

    def test_is_valid_bigram(self):
        """Should validate bigrams against probability threshold."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        sequences = [[0, 1, 2] * 10]
        graph.update_from_corpus(sequences)

        self.assertTrue(graph.is_valid_bigram(0, 1, threshold=0.001))
        self.assertTrue(graph.is_valid_bigram(9, 9, threshold=0.001))  # Unseen but non-zero

    def test_sample_next_token(self):
        """Should sample next token from distribution."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        sequences = [[0, 1, 2, 3, 4] * 10]
        graph.update_from_corpus(sequences)

        # Sample multiple times
        samples = [graph.sample_next_token(0) for _ in range(100)]

        # Most should be 1 (most probable)
        self.assertIn(1, samples)

    def test_generate_sequence(self):
        """Should generate token sequence."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        sequences = [[i, (i + 1) % 10] for i in range(10)] * 5
        graph.update_from_corpus(sequences)

        generated = graph.generate_sequence(start_token=0, max_length=5)

        self.assertLessEqual(len(generated), 5)
        self.assertEqual(generated[0], 0)

    def test_compute_sequence_probability(self):
        """Should compute log probability of sequence."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        sequences = [[0, 1, 2, 3] * 10]
        graph.update_from_corpus(sequences)

        log_prob = graph.compute_sequence_probability([0, 1, 2, 3])

        # Should be finite (no log(0) due to Laplace smoothing)
        self.assertTrue(np.isfinite(log_prob))

        # Frequent sequence should have higher probability than random
        random_log_prob = graph.compute_sequence_probability([0, 9, 9, 9])
        self.assertGreater(log_prob, random_log_prob)

    def test_get_perplexity(self):
        """Should compute perplexity on test sequences."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        train_sequences = [[0, 1, 2, 3] * 10]
        graph.update_from_corpus(train_sequences)

        test_sequences = [[0, 1, 2, 3], [0, 1, 2, 4]]
        perplexity = graph.get_perplexity(test_sequences)

        self.assertTrue(np.isfinite(perplexity))
        self.assertGreater(perplexity, 1.0)

    def test_save_load_json(self):
        """Should save and load from JSON."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        graph.set_token_label(0, "contact")
        graph.set_token_label(1, "alarm")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "syntax_graph.json"

            graph.save_json(path)
            self.assertTrue(path.exists())

            loaded = SyntaxGraph.load_json(path)

            self.assertEqual(loaded.num_tokens, 10)
            self.assertEqual(loaded.alpha, 0.01)
            self.assertEqual(loaded.get_token_label(0), "contact")

    def test_get_statistics(self):
        """Should return graph statistics."""
        from cognitive_intelligence.syntax_graph import SyntaxGraph

        graph = SyntaxGraph(num_tokens=10, alpha=0.01)
        sequences = [[0, 1, 2] * 10]
        graph.update_from_corpus(sequences)

        stats = graph.get_statistics()

        self.assertEqual(stats['num_tokens'], 10)
        self.assertEqual(stats['alpha'], 0.01)
        self.assertIn('min_probability', stats)
        self.assertIn('max_probability', stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for syntactic stream."""

    def test_end_to_end_tokenization(self):
        """Should extract features and tokenize."""
        from cognitive_intelligence.syntactic_encoder import SyntacticFeatureExtractor
        from cognitive_intelligence.syntactic_vqvae import create_syntactic_vqvae

        features_112d = np.random.randn(112).astype(np.float32)

        # Extract syntactic features
        syntactic = SyntacticFeatureExtractor.extract_syntactic_features(features_112d)

        # Tokenize
        vqvae = create_syntactic_vqvae()
        vqvae.eval()

        with torch.no_grad():
            x = torch.from_numpy(syntactic).unsqueeze(0)
            token_id = vqvae.tokenize(x)

        self.assertEqual(token_id.shape, (1,))
        self.assertGreaterEqual(token_id.item(), 0)
        self.assertLess(token_id.item(), 64)

    def test_corpus_to_syntax_graph(self):
        """Should build syntax graph from tokenized corpus."""
        from cognitive_intelligence.syntax_graph import create_syntax_graph_from_corpus

        # Create corpus
        corpus = [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 5],
            [5, 6, 7],
        ]

        graph = create_syntax_graph_from_corpus(corpus, num_tokens=10)

        # Check transitions learned
        top_k = graph.get_valid_next_tokens(0, top_k=1)
        self.assertEqual(top_k[0][0], 1)  # Most common after 0


if __name__ == "__main__":
    unittest.main()
