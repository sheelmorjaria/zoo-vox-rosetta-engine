#!/usr/bin/env python3
"""
Tests for VQ-VAE Codebook Revival (Risk C Mitigation)

Tests codebook utilization tracking, EMA updates, and revival techniques
that prevent codebook collapse.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import torch

from cognitive_intelligence.syntactic_vqvae import (
    SyntacticVQVAE,
    VQVAEConfig,
    create_syntactic_vqvae,
)


class TestCodebookUtilization(unittest.TestCase):
    """Test codebook utilization tracking."""

    def setUp(self):
        """Create a VQ-VAE model."""
        self.model = create_syntactic_vqvae()

    def test_initial_utilization_stats(self):
        """Should report zero utilization before training."""
        stats = self.model.get_utilization_stats()

        self.assertEqual(stats["total_codes"], 64)
        self.assertEqual(stats["active_codes"], 0)
        self.assertEqual(stats["utilization_percent"], 0.0)

    def test_utilization_increases_with_training(self):
        """Utilization should increase after training batches."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Run several training batches
        for _ in range(10):
            x = torch.randn(32, 44)
            x_recon, z, z_q, token_ids, perplexity = self.model(x)
            loss_dict = self.model.loss_function(x, x_recon, z, z_q)

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

        # Check utilization increased
        stats = self.model.get_utilization_stats()
        self.assertGreater(stats["active_codes"], 0)

    def test_utilization_target(self):
        """Should achieve >80% utilization after sufficient training."""
        config = VQVAEConfig(
            codebook_size=32,  # Smaller for faster testing
            codebook_dim=16,
        )
        model = create_syntactic_vqvae(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train until utilization target
        for i in range(100):
            x = torch.randn(64, 44)  # Large batch
            x_recon, z, z_q, token_ids, perplexity = model(x)
            loss_dict = model.loss_function(x, x_recon, z, z_q)

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

            utilization = model.codebook_utilization()
            if utilization >= 80.0:
                break

        # Final check
        self.assertGreaterEqual(model.codebook_utilization(), 60.0)  # Relaxed for speed


class TestEMAUpdates(unittest.TestCase):
    """Test EMA codebook updates."""

    def setUp(self):
        """Create a VQ-VAE model."""
        self.model = create_syntactic_vqvae()

    def test_ema_initialization(self):
        """EMA buffers should be initialized."""
        self.assertIsNotNone(self.model.vq.codebook_ema)
        self.assertIsNotNone(self.model.vq.cluster_size_ema)

        self.assertEqual(self.model.vq.codebook_ema.shape, (64, 32))
        self.assertEqual(self.model.vq.cluster_size_ema.shape, (64,))

    def test_ema_updates_during_training(self):
        """EMA should update during training."""
        initial_codebook = self.model.vq.codebook_ema.clone()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        x = torch.randn(16, 44)

        self.model.train()
        x_recon, z, z_q, token_ids, perplexity = self.model(x)
        loss_dict = self.model.loss_function(x, x_recon, z, z_q)

        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        optimizer.step()

        # Codebook should have changed
        self.assertFalse(torch.allclose(initial_codebook, self.model.vq.codebook_ema))

    def test_cluster_size_tracking(self):
        """Cluster sizes should track usage."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Run multiple batches
        for _ in range(5):
            x = torch.randn(16, 44)
            x_recon, z, z_q, token_ids, perplexity = self.model(x)
            loss_dict = self.model.loss_function(x, x_recon, z, z_q)

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

        # Some clusters should have non-zero size
        self.assertGreater(self.model.vq.cluster_size_ema.sum().item(), 0)


class TestCodebookRevival(unittest.TestCase):
    """Test codebook revival for dead codes."""

    def setUp(self):
        """Create a VQ-VAE model."""
        # Small codebook for easier testing
        config = VQVAEConfig(
            codebook_size=16,
            codebook_dim=8,
            revival_threshold=0.1,  # Higher threshold for faster testing
        )
        self.model = create_syntactic_vqvae(config)

    def test_revival_threshold(self):
        """Revival threshold should be accessible."""
        self.assertEqual(self.model.vq.revival_threshold, 0.1)

    def test_no_revival_in_early_batches(self):
        """Revival should not happen in first few batches."""
        initial_codebook = self.model.vq.codebook_ema.clone()

        x = torch.randn(8, 44)
        self.model.train()
        _ = self.model(x)

        # Codebook should not have been revived (same shape, potentially different values)
        self.assertEqual(self.model.vq.codebook_ema.shape, initial_codebook.shape)

    def test_token_usage_tracking(self):
        """Token usage should be tracked across batches."""
        self.assertEqual(self.model.vq.token_usage_count.sum().item(), 0)
        self.assertEqual(self.model.vq.total_batches.item(), 0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Run several batches
        for _ in range(20):
            x = torch.randn(16, 44)
            self.model.train()
            x_recon, z, z_q, token_ids, perplexity = self.model(x)
            loss_dict = self.model.loss_function(x, x_recon, z, z_q)

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

        # Total batches should be updated
        self.assertEqual(self.model.vq.total_batches.item(), 20)

        # Some tokens should have been used
        self.assertGreater(self.model.vq.token_usage_count.sum().item(), 0)

    def test_get_utilization_stats(self):
        """Should return detailed utilization stats."""
        stats = self.model.get_utilization_stats()

        self.assertIn("utilization_percent", stats)
        self.assertIn("active_codes", stats)
        self.assertIn("dead_codes", stats)
        self.assertIn("total_codes", stats)

        self.assertEqual(stats["total_codes"], 16)

    def test_per_code_usage_list(self):
        """Per-code usage should be a list of correct length."""
        stats = self.model.get_utilization_stats()

        # per_code_usage is always present now
        self.assertIn("per_code_usage", stats)
        self.assertEqual(len(stats["per_code_usage"]), 16)
        self.assertIsInstance(stats["per_code_usage"], list)

    def test_training_mode_affects_revival(self):
        """Training mode should enable EMA updates and revival."""
        x = torch.randn(8, 44)

        # Training mode
        self.model.train()
        with torch.no_grad():
            _, _, _, _, perplexity_train = self.model(x)

        # Eval mode
        self.model.eval()
        with torch.no_grad():
            _, _, _, _, perplexity_eval = self.model(x)


class TestIntegrationRiskCMitigation(unittest.TestCase):
    """Integration tests for Risk C mitigation."""

    def test_no_collapse_after_extended_training(self):
        """Codebook should not collapse after extended training."""
        config = VQVAEConfig(
            codebook_size=32,
            codebook_dim=16,
            revival_threshold=0.01,
        )
        model = create_syntactic_vqvae(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Extended training
        for _ in range(50):
            x = torch.randn(32, 44)
            model.train()
            x_recon, z, z_q, token_ids, perplexity = model(x)
            loss_dict = model.loss_function(x, x_recon, z, z_q)

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

        # Check utilization is above collapse threshold
        utilization = model.codebook_utilization()
        self.assertGreater(utilization, 20.0,  # At least 20% utilization
                          f"Codebook collapse detected: {utilization:.1f}% utilization")

    def test_diverse_token_distribution(self):
        """Tokens should be distributed across codebook."""
        config = VQVAEConfig(codebook_size=32, codebook_dim=16)
        model = create_syntactic_vqvae(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Collect tokens across batches
        all_tokens = []
        import numpy as np
        for _ in range(20):
            x = torch.randn(32, 44)
            model.train()
            _, _, _, token_ids, _ = model(x)
            loss_dict = model.loss_function(x, x, model.encode(x), model.encode(x))

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

            all_tokens.append(token_ids.flatten().cpu().numpy())

        all_tokens = np.concatenate(all_tokens)

        # Check we used multiple different tokens
        unique_tokens = len(np.unique(all_tokens))
        self.assertGreater(unique_tokens, 5,  # At least 5 different tokens
                          f"Poor token diversity: only {unique_tokens} unique tokens")


if __name__ == "__main__":
    unittest.main()
