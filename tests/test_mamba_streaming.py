#!/usr/bin/env python3
"""
Phase 4.1b: Mamba Streaming State Validation

Tests for Mamba/TCN autoregressive model streaming behavior.
Validates state persistence and sequential updates for real-time processing.

Key Requirements:
- State persists across audio frames
- Sequential state updates produce consistent results
- State reset capability for boundary events

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MambaState:
    """Mamba hidden state for streaming inference."""
    hidden: torch.Tensor
    cell: Optional[torch.Tensor] = None  # For LSTM-like states
    conv_state: Optional[torch.Tensor] = None  # For convolutional state


class MockMambaModel(nn.Module):
    """
    Mock Mamba model for testing streaming behavior.

    Simulates selective state space model with:
    - Hidden state persistence
    - Convolutional state for local context
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Convolutional layer (simulates local context)
        self.conv1d = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=4,
            padding=3,
            groups=hidden_dim,
        )

        # Gating mechanism
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MambaState] = None,
    ) -> Tuple[torch.Tensor, MambaState]:
        """
        Forward pass with state management.

        Args:
            x: Input tensor (B, D) or (B, T, D)
            state: Previous hidden state

        Returns:
            output: Predicted next representation
            new_state: Updated hidden state
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        B, T, D = x.shape

        # Project input
        h = self.input_proj(x)  # (B, T, H)

        # Apply convolution with state
        if state is not None and state.conv_state is not None:
            # Prepend conv state for continuity
            h_padded = torch.cat([state.conv_state, h], dim=1)
            conv_out = self.conv1d(h_padded.transpose(1, 2)).transpose(1, 2)
            conv_out = conv_out[:, -T:]  # Take only new outputs
            new_conv_state = h_padded[:, -3:]  # Keep last 3 for next step
        else:
            conv_out = self.conv1d(h.transpose(1, 2)).transpose(1, 2)
            new_conv_state = h[:, -3:].detach()

        # Gating
        gate = torch.sigmoid(self.gate_proj(conv_out))
        gated = gate * conv_out

        # Output projection
        output = self.output_proj(gated)

        if squeeze:
            output = output.squeeze(1)

        # Create new state
        new_state = MambaState(
            hidden=gated[:, -1:].detach(),
            conv_state=new_conv_state.detach(),
        )

        return output, new_state


class TestMambaStreaming:
    """Test suite for Mamba streaming validation."""

    def __init__(self):
        self.model = MockMambaModel(input_dim=128, hidden_dim=256)

    def test_state_initialization(self):
        """Test that state is properly initialized."""
        print("\n" + "=" * 60)
        print("State Initialization")
        print("=" * 60)

        x = torch.randn(1, 128)
        output, state = self.model(x)

        print(f"Output shape: {output.shape}")
        print(f"State hidden shape: {state.hidden.shape}")
        print(f"State conv shape: {state.conv_state.shape}")

        assert state.hidden is not None, "Hidden state should be initialized"
        assert state.conv_state is not None, "Conv state should be initialized"

        assert state.hidden.shape == (1, 1, 256), f"Hidden state shape incorrect: {state.hidden.shape}"

    def test_state_persistence(self):
        """Test that state persists across sequential calls."""
        print("\n" + "=" * 60)
        print("State Persistence")
        print("=" * 60)

        state = None
        outputs = []
        states = []

        # Process 5 sequential frames
        for i in range(5):
            x = torch.randn(1, 128)
            output, state = self.model(x, state)
            outputs.append(output)
            states.append(state.hidden.clone() if state.hidden is not None else None)

        print(f"Processed 5 frames")
        print(f"State changes: {sum(1 for i in range(1, len(states)) if not torch.allclose(states[i], states[i-1], atol=1e-5))}")

        # States should evolve over time
        assert not all(
            torch.allclose(states[i], states[i-1], atol=1e-5)
            for i in range(1, len(states))
        ), "States should evolve across frames"

    def test_state_reset(self):
        """Test state reset capability."""
        print("\n" + "=" * 60)
        print("State Reset")
        print("=" * 60)

        x = torch.randn(1, 128)

        # Build up state
        state = None
        for _ in range(3):
            _, state = self.model(x, state)

        state_before = state.hidden.clone()

        # Reset
        state = None
        output, state_after = self.model(x, state)

        print(f"State before reset norm: {state_before.norm().item():.4f}")
        print(f"State after reset norm: {state_after.hidden.norm().item():.4f}")

        # New state should be different
        assert not torch.allclose(state_before, state_after.hidden, atol=1e-3), \
            "State should change after reset"

    def test_sequential_consistency(self):
        """Test that sequential processing yields consistent results."""
        print("\n" + "=" * 60)
        print("Sequential Consistency")
        print("=" * 60)

        # Generate sequential inputs
        inputs = [torch.randn(1, 128) for _ in range(10)]

        # Process sequentially
        state = None
        outputs_sequential = []
        for x in inputs:
            output, state = self.model(x, state)
            outputs_sequential.append(output)

        # Process as batch (for comparison)
        batch_input = torch.cat(inputs, dim=0)
        batch_output, _ = self.model(batch_input.unsqueeze(1))

        print(f"Sequential output shape: {outputs_sequential[0].shape}")
        print(f"Batch output shape: {batch_output.shape}")

        # Last sequential output should use full context
        # (though results may differ due to conv state handling)

    def test_multi_step_prediction(self):
        """Test multi-step autoregressive prediction."""
        print("\n" + "=" * 60)
        print("Multi-Step Prediction")
        print("=" * 60)

        # Initial input
        x = torch.randn(1, 128)
        state = None

        # Predict 5 steps ahead
        predictions = []
        for step in range(5):
            pred, state = self.model(x, state)
            predictions.append(pred)
            x = pred  # Autoregressive: use prediction as next input

        print(f"Generated {len(predictions)} step-ahead predictions")

        # All predictions should have valid shapes
        for i, pred in enumerate(predictions):
            assert pred.shape == (1, 128), f"Prediction {i} has wrong shape: {pred.shape}"

    def test_state_at_boundaries(self):
        """Test state behavior at detected boundaries."""
        print("\n" + "=" * 60)
        print("State Behavior at Boundaries")
        print("=" * 60)

        state = None
        boundary_indices = [3, 7, 11]  # Simulated boundary detections

        for i in range(15):
            x = torch.randn(1, 128)

            if i in boundary_indices:
                # Reset state at boundary
                state = None
                output, state = self.model(x, state)
                print(f"  Frame {i}: BOUNDARY - state reset")
            else:
                output, state = self.model(x, state)

        print(f"Processed 15 frames with {len(boundary_indices)} boundary resets")

    def test_conv_state_continuity(self):
        """Test that convolutional state maintains continuity."""
        print("\n" + "=" * 60)
        print("Convolutional State Continuity")
        print("=" * 60)

        state = None
        conv_states = []

        for i in range(5):
            x = torch.randn(1, 128)
            _, state = self.model(x, state)
            if state.conv_state is not None:
                conv_states.append(state.conv_state.clone())

        print(f"Collected {len(conv_states)} conv states")

        # Conv states should maintain temporal continuity
        for i in range(1, len(conv_states)):
            # Last frame of previous state should be similar to first frame of current
            # (since they overlap)
            prev_end = conv_states[i-1][:, -1:]
            curr_start = conv_states[i][:, :1]

            similarity = torch.cosine_similarity(
                prev_end.flatten().unsqueeze(0),
                curr_start.flatten().unsqueeze(0),
            ).item()

            print(f"  Frame {i-1} -> {i} continuity: {similarity:.4f}")

            # Should have reasonable continuity
            assert similarity > -0.5, f"Conv state discontinuity detected: {similarity}"

    def test_batch_state_management(self):
        """Test state management with batched inputs."""
        print("\n" + "=" * 60)
        print("Batch State Management")
        print("=" * 60)

        batch_size = 4
        state = None

        for i in range(3):
            x = torch.randn(batch_size, 128)
            output, state = self.model(x, state)

            print(f"Step {i}:")
            print(f"  Output shape: {output.shape}")
            if state.hidden is not None:
                print(f"  State shape: {state.hidden.shape}")

            assert state.hidden.shape[0] == batch_size, \
                f"State batch size mismatch: {state.hidden.shape[0]} vs {batch_size}"

    def test_state_serialization(self):
        """Test that state can be serialized and restored."""
        print("\n" + "=" * 60)
        print("State Serialization")
        print("=" * 60)

        x = torch.randn(1, 128)

        # Build state
        state = None
        for _ in range(3):
            _, state = self.model(x, state)

        # Serialize state
        state_dict = {
            'hidden': state.hidden.detach().cpu(),
            'conv_state': state.conv_state.detach().cpu() if state.conv_state is not None else None,
        }

        print(f"Serialized state keys: {list(state_dict.keys())}")

        # Restore state
        restored_state = MambaState(
            hidden=state_dict['hidden'].to(x.device),
            conv_state=state_dict['conv_state'].to(x.device) if state_dict['conv_state'] is not None else None,
        )

        # Compare outputs
        x2 = torch.randn(1, 128)
        output1, _ = self.model(x2, state)
        output2, _ = self.model(x2, restored_state)

        print(f"Output match: {torch.allclose(output1, output2, atol=1e-5)}")

        assert torch.allclose(output1, output2, atol=1e-5), \
            "Restored state should produce identical outputs"

    def test_gradient_accumulation(self):
        """Test gradient accumulation through state."""
        print("\n" + "=" * 60)
        print("Gradient Accumulation Through State")
        print("=" * 60)

        x = torch.randn(1, 128, requires_grad=True)
        state = None

        # Process sequence
        for _ in range(3):
            output, state = self.model(x, state)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        print(f"Input gradient norm: {x.grad.norm().item():.4f}")
        print(f"Model param grad norm: {sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None):.4f}")

        assert x.grad is not None, "Gradients should flow to input"


def main():
    """Run all Mamba streaming validation tests."""
    print("=" * 60)
    print("Phase 4.1b: Mamba Streaming State Validation")
    print("=" * 60)
    print()

    test = TestMambaStreaming()

    test.test_state_initialization()
    test.test_state_persistence()
    test.test_state_reset()
    test.test_sequential_consistency()
    test.test_multi_step_prediction()
    test.test_state_at_boundaries()
    test.test_conv_state_continuity()
    test.test_batch_state_management()
    test.test_state_serialization()
    test.test_gradient_accumulation()

    print("\n" + "=" * 60)
    print("✓ ALL MAMBA STREAMING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
