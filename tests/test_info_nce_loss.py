#!/usr/bin/env python3
"""
Phase 4.1a: InfoNCE Loss Validation

Tests for the InfoNCE (Contrastive Predictive Coding) loss function
used in Predictive NBD for mutual information maximization.

InfoNCE Loss = -log(exp(sim(z_k, z_q)/τ) / Σ_j exp(sim(z_j, z_q)/τ))

Where:
- z_k: Positive sample (true future)
- z_q: Query (current representation)
- z_j: Negative samples (distractors)
- τ: Temperature parameter

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Contrastive Predictive Coding.

    Maximizes mutual information between current representation
    and predicted future representations.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        queries: torch.Tensor,      # (B, N, D)
        positive_keys: torch.Tensor,  # (B, N, D)
        negative_keys: torch.Tensor,  # (B, N, K, D) or (B, N, D)
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            queries: Current representations (B, N, D)
            positive_keys: True future representations (B, N, D)
            negative_keys: Distractor representations (B, N, K, D) or (B, N, D)

        Returns:
            Scalar loss value
        """
        B, N, D = queries.shape

        # Flatten batch and sequence dimensions
        queries_flat = queries.view(-1, D)  # (B*N, D)
        positive_flat = positive_keys.view(-1, D)  # (B*N, D)

        # Compute positive similarities
        pos_sim = torch.sum(queries_flat * positive_flat, dim=-1) / self.temperature  # (B*N,)

        # Compute negative similarities
        if negative_keys.dim() == 4:
            # (B, N, K, D) -> (B*N, K, D)
            neg_keys_flat = negative_keys.view(B * N, -1, D)
            # Compute similarities against all negatives
            neg_sim = torch.einsum('bd,bkd->bk', queries_flat, neg_keys_flat) / self.temperature
        else:
            # Single negative key
            neg_keys_flat = negative_keys.view(-1, D)
            neg_sim = torch.sum(queries_flat * neg_keys_flat, dim=-1, keepdim=True) / self.temperature

        # Concatenate positive and negative similarities
        # Shape: (B*N, 1+K) or (B*N, 2)
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        # Compute cross-entropy loss (positive is index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_mutual_information(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate mutual information I(Z_q; Z_k).

        MI ≥ H(Z_q) - H(Z_q|Z_k) ≈ log(N) - L
        where L is the InfoNCE loss.

        Args:
            queries: (B, N, D)
            keys: (B, N, D)

        Returns:
            Estimated mutual information (scalar)
        """
        B, N, D = queries.shape
        num_codes = N  # Number of discrete codes

        # Compute InfoNCE loss
        loss = self.forward(queries, keys, keys)

        # Lower bound on MI: I ≥ log(N) - L
        mi_lower_bound = torch.log(torch.tensor(num_codes, dtype=loss.dtype)) - loss

        return mi_lower_bound


class TestInfoNCELoss:
    """Test suite for InfoNCE loss validation."""

    def __init__(self):
        self.loss_fn = InfoNCELoss(temperature=0.1)

    def test_info_nce_loss_computation(self):
        """Test basic InfoNCE loss computation."""
        print("\n" + "=" * 60)
        print("InfoNCE Loss Computation")
        print("=" * 60)

        # Create test data
        B, N, D, K = 4, 8, 64, 10

        queries = torch.randn(B, N, D)
        positive_keys = torch.randn(B, N, D)
        negative_keys = torch.randn(B, N, K, D)

        # Compute loss
        loss = self.loss_fn(queries, positive_keys, negative_keys)

        print(f"InfoNCE loss: {loss.item():.4f}")

        # Loss should be positive
        assert loss.item() > 0, "InfoNCE loss should be positive"

        # Loss should be finite
        assert torch.isfinite(loss), "InfoNCE loss should be finite"

    def test_loss_with_perfect_predictions(self):
        """Test loss with identical queries and keys (lower bound)."""
        print("\n" + "=" * 60)
        print("Loss with Perfect Predictions")
        print("=" * 60)

        B, N, D = 4, 8, 64

        # Identical queries and positive keys
        queries = torch.randn(B, N, D)
        positive_keys = queries.clone()
        negative_keys = torch.randn(B, N, 10, D)

        loss = self.loss_fn(queries, positive_keys, negative_keys)

        print(f"Loss with perfect match: {loss.item():.4f}")

        # Loss should be low (close to log(1/(1+K)))
        # Lower bound: -log(1/(1+K)) = log(1+K)
        expected_lower = np.log(1 + 10)
        print(f"Theoretical lower bound: {expected_lower:.4f}")

        # Actual loss should be less than expected for random
        assert loss.item() < expected_lower + 0.5, \
            "Perfect predictions should yield low loss"

    def test_loss_with_random_predictions(self):
        """Test loss with uncorrelated queries and keys."""
        print("\n" + "=" * 60)
        print("Loss with Random Predictions")
        print("=" * 60)

        B, N, D, K = 4, 8, 64, 10

        queries = torch.randn(B, N, D)
        positive_keys = torch.randn(B, N, D)
        negative_keys = torch.randn(B, N, K, D)

        loss = self.loss_fn(queries, positive_keys, negative_keys)

        print(f"Loss with random predictions: {loss.item():.4f}")

        # Random predictions should yield loss near log(1+K)
        expected = np.log(1 + K)
        print(f"Expected (random): {expected:.4f}")

        # Should be close to expected
        assert abs(loss.item() - expected) < 1.0, \
            f"Random loss {loss.item():.4f} should be near {expected:.4f}"

    def test_mutual_information_estimation(self):
        """Test mutual information lower bound estimation."""
        print("\n" + "=" * 60)
        print("Mutual Information Estimation")
        print("=" * 60)

        B, N, D = 4, 8, 64

        # Correlated queries and keys
        queries = torch.randn(B, N, D)
        positive_keys = queries + 0.1 * torch.randn(B, N, D)  # Add small noise

        mi_estimate = self.loss_fn.compute_mutual_information(queries, positive_keys)

        print(f"MI lower bound estimate: {mi_estimate.item():.4f} nats")
        print(f"MI lower bound estimate: {mi_estimate.item() / np.log(2):.4f} bits")

        # MI should be positive for correlated variables
        assert mi_estimate.item() > 0, \
            "MI estimate should be positive for correlated variables"

    def test_temperature_sensitivity(self):
        """Test effect of temperature parameter on loss."""
        print("\n" + "=" * 60)
        print("Temperature Sensitivity")
        print("=" * 60)

        B, N, D, K = 4, 8, 64, 10

        queries = torch.randn(B, N, D)
        positive_keys = torch.randn(B, N, D)
        negative_keys = torch.randn(B, N, K, D)

        temperatures = [0.01, 0.1, 0.5, 1.0]

        for temp in temperatures:
            loss_fn = InfoNCELoss(temperature=temp)
            loss = loss_fn(queries, positive_keys, negative_keys)
            print(f"  Temperature {temp}: loss = {loss.item():.4f}")

        # Lower temperature = sharper distribution = different loss

    def test_negative_sampling_impact(self):
        """Test effect of number of negative samples."""
        print("\n" + "=" * 60)
        print("Negative Sampling Impact")
        print("=" * 60)

        B, N, D = 4, 8, 64

        queries = torch.randn(B, N, D)
        positive_keys = torch.randn(B, N, D)

        K_values = [1, 5, 10, 50, 100]

        for K in K_values:
            negative_keys = torch.randn(B, N, K, D)
            loss = self.loss_fn(queries, positive_keys, negative_keys)
            print(f"  K={K:3d}: loss = {loss.item():.4f}")

        # More negatives = harder task = typically higher loss

    def test_gradient_flow(self):
        """Test that gradients flow properly through loss."""
        print("\n" + "=" * 60)
        print("Gradient Flow Test")
        print("=" * 60)

        B, N, D, K = 4, 8, 64, 10

        # Create trainable query encoder
        encoder = nn.Linear(D, D)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)

        queries = torch.randn(B, N, D, requires_grad=True)
        positive_keys = torch.randn(B, N, D)
        negative_keys = torch.randn(B, N, K, D)

        # Forward and backward
        encoded = encoder(queries)
        loss = self.loss_fn(encoded, positive_keys, negative_keys)
        loss.backward()

        # Check gradients exist
        assert queries.grad is not None, "Queries should have gradients"
        assert encoder.weight.grad is not None, "Encoder weights should have gradients"

        print(f"Query gradient norm: {queries.grad.norm().item():.4f}")
        print(f"Encoder weight grad norm: {encoder.weight.grad.norm().item():.4f}")

        # Gradients should be non-zero
        assert queries.grad.norm().item() > 0, "Query gradients should be non-zero"

    def test_batch_and_sequence_dimensions(self):
        """Test loss with various batch and sequence sizes."""
        print("\n" + "=" * 60)
        print("Batch and Sequence Dimension Tests")
        print("=" * 60)

        configs = [
            (1, 1, 32, 5),   # Single sample, single step
            (2, 4, 64, 10),  # Small batch
            (8, 16, 128, 20), # Larger batch
        ]

        for B, N, D, K in configs:
            queries = torch.randn(B, N, D)
            positive_keys = torch.randn(B, N, D)
            negative_keys = torch.randn(B, N, K, D)

            loss = self.loss_fn(queries, positive_keys, negative_keys)

            print(f"  B={B}, N={N}, D={D}, K={K}: loss = {loss.item():.4f}")

            assert torch.isfinite(loss), f"Loss should be finite for B={B}, N={N}"


def main():
    """Run all InfoNCE validation tests."""
    print("=" * 60)
    print("Phase 4.1a: InfoNCE Loss Validation")
    print("=" * 60)
    print()

    test = TestInfoNCELoss()

    test.test_info_nce_loss_computation()
    test.test_loss_with_perfect_predictions()
    test.test_loss_with_random_predictions()
    test.test_mutual_information_estimation()
    test.test_temperature_sensitivity()
    test.test_negative_sampling_impact()
    test.test_gradient_flow()
    test.test_batch_and_sequence_dimensions()

    print("\n" + "=" * 60)
    print("✓ ALL INFONCE VALIDATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
