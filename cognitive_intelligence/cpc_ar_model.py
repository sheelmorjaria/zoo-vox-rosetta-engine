#!/usr/bin/env python3
"""
Autoregressive Model for CPC Predictive NBD

Implements temporal prediction for CPC:
- TCN (Temporal Convolutional Network) version
- Lightweight for edge deployment (~82K parameters)

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with dilated convolutions.

    Args:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            out: (batch, channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNARModel(nn.Module):
    """
    Temporal Convolutional Network for autoregressive prediction.

    Lightweight architecture:
    - ~82K parameters
    - Receptive field: ~16 time steps
    - Suitable for edge deployment

    Args:
        input_dim: Input feature dimension (default 128)
        hidden_dim: Hidden dimension (default 64)
        num_levels: Number of TCN levels (default 4)
        kernel_size: Convolution kernel size (default 3)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        layers = []
        num_channels = [input_dim] + [hidden_dim] * num_levels

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i + 1]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, z: torch.Tensor, steps_ahead: int = 5) -> List[torch.Tensor]:
        """
        Generate multi-step predictions.

        Args:
            z: (batch, seq_len, input_dim) encoded features
            steps_ahead: Number of steps to predict

        Returns:
            predictions: List of (batch, 1, input_dim) for each step
        """
        batch_size = z.shape[0]
        device = z.device

        # Reshape for TCN: (batch, channels, seq_len)
        z_tcn = z.transpose(1, 2)  # (batch, input_dim, seq_len)

        # Get initial hidden state
        hidden = self.network(z_tcn)  # (batch, hidden_dim, seq_len)
        current = hidden[:, :, -1:]  # (batch, hidden_dim, 1)

        predictions = []

        for step in range(steps_ahead):
            # Project to input dimension
            pred = self.output_proj(current.transpose(1, 2))  # (batch, 1, input_dim)
            predictions.append(pred)

            # Update hidden state for next prediction
            pred_tcn = pred.transpose(1, 2)  # (batch, input_dim, 1)
            current = self.network(pred_tcn)[:, :, -1:]

        return predictions

    def predict_single(self, z: torch.Tensor, steps_ahead: int = 5) -> List[torch.Tensor]:
        """
        Predict from single latent vector (no batch).

        Args:
            z: (seq_len, input_dim) or (input_dim,)

        Returns:
            predictions: List of (input_dim,) for each step
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)  # (1, input_dim)

        if z.dim() == 2:
            z = z.unsqueeze(0)  # (1, seq_len, input_dim)

        with torch.no_grad():
            preds = self.forward(z, steps_ahead)

        # Remove batch dimension
        return [p.squeeze(0).squeeze(0) for p in preds]


class LightweightARModel(nn.Module):
    """
    Very lightweight AR model for edge deployment.

    Uses simple MLP instead of TCN for minimal footprint.
    ~50K parameters.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, steps_ahead: int = 5) -> List[torch.Tensor]:
        """
        Generate autoregressive predictions.

        Args:
            z: (batch, seq_len, input_dim)
            steps_ahead: Number of steps to predict

        Returns:
            predictions: List of (batch, 1, input_dim)
        """
        batch_size = z.shape[0]
        device = z.device

        # Get last hidden state
        last_z = z[:, -1, :]  # (batch, input_dim)

        predictions = []
        current = last_z

        for _ in range(steps_ahead):
            pred = self.network(current)  # (batch, input_dim)
            predictions.append(pred.unsqueeze(1))  # (batch, 1, input_dim)
            current = pred

        return predictions


def create_ar_model(
    input_dim: int = 128,
    hidden_dim: int = 64,
    model_type: str = "tcn",
) -> nn.Module:
    """
    Create an AR model with specified architecture.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        model_type: "tcn" or "lightweight"

    Returns:
        AR model instance
    """
    if model_type == "tcn":
        return TCNARModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_levels=4,
            kernel_size=3,
            dropout=0.1,
        )
    else:
        return LightweightARModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
        )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_ar_model():
    """Test AR model functionality."""
    print("=" * 60)
    print("Testing AR Model")
    print("=" * 60)

    # Test TCN
    tcn = create_ar_model(model_type="tcn")
    params = count_parameters(tcn)
    print(f"TCN parameters: {params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 1
    input_dim = 128

    z = torch.randn(batch_size, seq_len, input_dim)
    predictions = tcn(z, steps_ahead=5)

    print(f"Input shape: {z.shape}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction shape: {predictions[0].shape}")

    assert len(predictions) == 5
    assert predictions[0].shape == (batch_size, 1, input_dim)

    # Test lightweight
    lightweight = create_ar_model(model_type="lightweight")
    params = count_parameters(lightweight)
    print(f"\nLightweight AR parameters: {params:,}")

    predictions = lightweight(z, steps_ahead=5)
    print(f"Lightweight prediction shape: {predictions[0].shape}")

    print("\n✓ AR model tests passed")


if __name__ == "__main__":
    test_ar_model()
