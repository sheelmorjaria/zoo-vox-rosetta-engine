#!/usr/bin/env python3
"""
Multi-Scale Spectral Loss - Module 2 (v1.6.0)

Differentiable loss function for training DDSP models.

Computes spectral distance between predicted and target audio at multiple
STFT resolutions, capturing both fine spectral details and coarse spectral
envelope. This is the standard loss function used in DDSP training.

The loss combines:
1. L1 spectral distance at multiple frame lengths
2. L2 spectral distance at multiple frame lengths
3. Optional perceptual weighting (emphasis on perceptually relevant frequencies)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International

References:
    - Engel et al. (2020) "DDSP: Differentiable Digital Signal Processing"
    - Kumar et al. (2019) "High-Fidelity Audio Generation with Fewer Labels"
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MultiScaleSpectralLossConfig:
    """Configuration for MultiScaleSpectralLoss."""

    # STFT frame lengths for multi-scale analysis
    frame_lengths: List[int] = (512, 1024, 2048, 4096)

    # Hop size as fraction of frame length
    hop_fraction: float = 0.25

    # Loss weights
    l1_weight: float = 1.0
    l2_weight: float = 1.0

    # Whether to use magnitude-only (vs complex)
    magnitude_only: bool = True

    # Sample rate (for frequency weighting)
    sample_rate: int = 48000

    # Perceptual weighting (emphasis on 2-8kHz range for animal vocalizations)
    perceptual_weighting: bool = False

    # Minimum loss floor (prevents division by zero)
    eps: float = 1e-8


# =============================================================================
# Multi-Scale Spectral Loss
# =============================================================================


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale STFT loss for audio generation.

    Computes spectral reconstruction loss at multiple time-frequency resolutions.
    This captures both:
    - Fine spectral details (short frames)
    - Coarse spectral envelope (long frames)

    The loss is differentiable, enabling gradient-based training of
    neural audio synthesizers.

    Example:
        >>> loss_fn = MultiScaleSpectralLoss()
        >>> pred_audio = torch.randn(1, 1, 16000)  # 1 second at 16kHz
        >>> target_audio = torch.randn(1, 1, 16000)
        >>> loss = loss_fn(pred_audio, target_audio)
        >>> print(loss.item())  # Scalar loss value
    """

    def __init__(
        self,
        frame_lengths: List[int] = (512, 1024, 2048, 4096),
        hop_fraction: float = 0.25,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        magnitude_only: bool = True,
        sample_rate: int = 48000,
        perceptual_weighting: bool = False,
    ):
        """
        Initialize MultiScaleSpectralLoss.

        Args:
            frame_lengths: List of FFT frame lengths for multi-scale analysis
            hop_fraction: Hop size as fraction of frame length
            l1_weight: Weight for L1 loss component
            l2_weight: Weight for L2 loss component
            magnitude_only: Use magnitude only (vs complex spectrogram)
            sample_rate: Audio sample rate in Hz
            perceptual_weighting: Apply perceptual frequency weighting
        """
        super().__init__()

        self.frame_lengths = frame_lengths
        self.hop_fraction = hop_fraction
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.magnitude_only = magnitude_only
        self.sample_rate = sample_rate
        self.perceptual_weighting = perceptual_weighting
        self.eps = 1e-8

        logger.info(
            f"MultiScaleSpectralLoss initialized with "
            f"frame_lengths={frame_lengths}, "
            f"l1_weight={l1_weight}, l2_weight={l2_weight}"
        )

    def stft(
        self,
        audio: torch.Tensor,
        frame_length: int,
    ) -> torch.Tensor:
        """
        Compute STFT at specified frame length.

        Args:
            audio: Input audio tensor (B, 1, T) or (B, T)
            frame_length: FFT frame length

        Returns:
            Spectrogram: (B, freq_bins, time_frames) if magnitude_only
                        (B, 2, freq_bins, time_frames) if complex
        """
        # Ensure 3D input (B, 1, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio.shape[0]
        sample_length = audio.shape[2]

        # Compute hop size
        hop_size = int(frame_length * self.hop_fraction)

        # Pad to ensure exact number of frames
        n_frames = (sample_length - frame_length) // hop_size + 1
        pad_length = (n_frames - 1) * hop_size + frame_length - sample_length

        if pad_length > 0:
            audio = F.pad(audio, (0, pad_length))

        # Compute STFT
        # Return shape: (B, n_fft//2 + 1, n_frames) for magnitude
        spec = torch.stft(
            audio.squeeze(1),  # (B, T)
            n_fft=frame_length,
            hop_length=hop_size,
            win_length=frame_length,
            window=torch.hann_window(frame_length, device=audio.device),
            return_complex=True,
        )  # (B, freq_bins, time_frames)

        if self.magnitude_only:
            # Magnitude spectrogram
            spec = spec.abs()
        else:
            # Complex spectrogram as (real, imag)
            spec = torch.stack([spec.real, spec.imag], dim=1)  # (B, 2, freq_bins, time_frames)

        return spec

    def perceptual_weight(
        self,
        spec: torch.Tensor,
        frame_length: int,
    ) -> torch.Tensor:
        """
        Apply perceptual frequency weighting.

        Emphasizes the 2-8kHz range which is perceptually important
        for animal vocalizations (and human speech).

        Args:
            spec: Spectrogram (B, freq_bins, time_frames) or (B, 2, freq_bins, time_frames)
            frame_length: FFT frame length (for computing frequency bins)

        Returns:
            Weighted spectrogram
        """
        if self.perceptual_weighting:
            # Compute frequency bin centers
            freq_bins = spec.shape[-2]
            freqs = torch.linspace(0, self.sample_rate / 2, freq_bins, device=spec.device)

            # Create weight curve: emphasize 2-8kHz range
            # Use a bandpass curve centered at 4kHz
            center_freq = 4000.0
            bandwidth = 4000.0
            weight = torch.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth**2))

            # Boost the emphasized range
            weight = 1.0 + 2.0 * weight  # Range: [1.0, 3.0]

            # Reshape for broadcasting
            if spec.dim() == 4:  # (B, 2, freq_bins, time_frames)
                weight = weight.view(1, 1, -1, 1)
            else:  # (B, freq_bins, time_frames)
                weight = weight.view(1, -1, 1)

            return spec * weight
        else:
            return spec

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale spectral loss.

        Args:
            pred_audio: Predicted audio (B, 1, T) or (B, T)
            target_audio: Target audio (B, 1, T) or (B, T)

        Returns:
            Scalar loss value
        """
        total_loss = 0.0
        total_weight = 0.0

        # Ensure both inputs have the same length
        min_length = min(pred_audio.shape[-1], target_audio.shape[-1])
        pred_audio = pred_audio[..., :min_length]
        target_audio = target_audio[..., :min_length]

        for frame_length in self.frame_lengths:
            # Skip if audio is shorter than frame length
            if min_length < frame_length:
                continue

            # Compute STFTs at this scale
            pred_spec = self.stft(pred_audio, frame_length)
            target_spec = self.stft(target_audio, frame_length)

            # Apply perceptual weighting if enabled
            pred_spec = self.perceptual_weight(pred_spec, frame_length)
            target_spec = self.perceptual_weight(target_spec, frame_length)

            # Ensure same shape (crop to minimum time frames)
            min_frames = min(pred_spec.shape[-1], target_spec.shape[-1])
            pred_spec = pred_spec[..., :min_frames]
            target_spec = target_spec[..., :min_frames]

            # L1 loss (magnitude distance)
            l1_loss = (pred_spec - target_spec).abs().mean()

            # L2 loss (spectral MSE)
            l2_loss = ((pred_spec - target_spec) ** 2).mean()

            # Combine losses
            scale_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss

            # Accumulate (normalize by number of scales)
            total_loss += scale_loss
            total_weight += 1.0

        return total_loss / max(total_weight, 1.0)


# =============================================================================
# Loss Variants
# =============================================================================


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral convergence loss (for training stability).

    Measures the angle between predicted and target spectra,
    which is more robust to overall magnitude differences.
    """

    def __init__(self, frame_length: int = 2048):
        super().__init__()
        self.frame_length = frame_length

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral convergence loss."""
        # Compute magnitude spectra
        pred_spec = torch.stft(
            pred_audio.squeeze() if pred_audio.dim() == 3 else pred_audio,
            n_fft=self.frame_length,
            return_complex=True,
        ).abs()

        target_spec = torch.stft(
            target_audio.squeeze() if target_audio.dim() == 3 else target_audio,
            n_fft=self.frame_length,
            return_complex=True,
        ).abs()

        # Cosine similarity
        pred_flat = pred_spec.flatten()
        target_flat = target_spec.flatten()

        dot = (pred_flat * target_flat).sum()
        norm_pred = (pred_flat**2).sum().sqrt()
        norm_target = (target_flat**2).sum().sqrt()

        # Convergence: 1 - cosine similarity
        convergence = 1.0 - (dot / (norm_pred * norm_target + 1e-8))
        return convergence


class TimeDomainLoss(nn.Module):
    """
    Time-domain loss component (complements spectral loss).

    Combines L1 distance in time domain with optional perceptual
    weighting.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Compute time-domain loss."""
        # Ensure same length
        min_length = min(pred_audio.shape[-1], target_audio.shape[-1])
        pred = pred_audio[..., :min_length]
        target = target_audio[..., :min_length]

        l1_loss = (pred - target).abs().mean()
        l2_loss = ((pred - target) ** 2).mean()

        return self.l1_weight * l1_loss + self.l2_weight * l2_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for DDSP training.

    Multi-scale spectral loss + time-domain loss for robust training.
    """

    def __init__(
        self,
        spectral_weight: float = 1.0,
        time_weight: float = 0.1,
        **spectral_kwargs,
    ):
        """
        Initialize combined loss.

        Args:
            spectral_weight: Weight for spectral loss component
            time_weight: Weight for time-domain loss component
            **spectral_kwargs: Arguments passed to MultiScaleSpectralLoss
        """
        super().__init__()

        self.spectral_weight = spectral_weight
        self.time_weight = time_weight

        self.spectral_loss = MultiScaleSpectralLoss(**spectral_kwargs)
        self.time_loss = TimeDomainLoss()

        logger.info(
            f"CombinedLoss initialized: "
            f"spectral_weight={spectral_weight}, time_weight={time_weight}"
        )

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss with individual components.

        Args:
            pred_audio: Predicted audio
            target_audio: Target audio

        Returns:
            Total loss and dictionary of individual loss components
        """
        spectral = self.spectral_loss(pred_audio, target_audio)
        time = self.time_loss(pred_audio, target_audio)

        total = self.spectral_weight * spectral + self.time_weight * time

        losses = {
            "total": total.item(),
            "spectral": spectral.item(),
            "time": time.item(),
        }

        return total, losses


# =============================================================================
# Utility Functions
# =============================================================================


def create_loss_fn(
    loss_type: str = "multiscale_spectral",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss function
            - "multiscale_spectral": Multi-scale STFT loss
            - "spectral_convergence": Spectral convergence loss
            - "time_domain": Time-domain L1/L2 loss
            - "combined": Combined spectral + time loss
        **kwargs: Arguments passed to loss constructor

    Returns:
        Loss function module
    """
    if loss_type == "multiscale_spectral":
        return MultiScaleSpectralLoss(**kwargs)
    elif loss_type == "spectral_convergence":
        return SpectralConvergenceLoss(**kwargs)
    elif loss_type == "time_domain":
        return TimeDomainLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample audio
    batch_size = 2
    sample_rate = 48000
    duration_sec = 0.5
    n_samples = sample_rate * duration_sec

    pred_audio = torch.randn(batch_size, n_samples)
    target_audio = torch.randn(batch_size, n_samples)

    print("\n=== Multi-Scale Spectral Loss Test ===")
    print(f"Batch size: {batch_size}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration_sec} sec")
    print(f"Samples: {n_samples}")

    # Test multi-scale spectral loss
    loss_fn = MultiScaleSpectralLoss()
    loss = loss_fn(pred_audio, target_audio)
    print(f"\nMulti-scale spectral loss: {loss.item():.6f}")

    # Test combined loss
    combined_fn = CombinedLoss()
    total, losses = combined_fn(pred_audio, target_audio)
    print("\n=== Combined Loss ===")
    print(f"Total: {losses['total']:.6f}")
    print(f"Spectral: {losses['spectral']:.6f}")
    print(f"Time: {losses['time']:.6f}")

    # Test with single sample
    print("\n=== Single Sample Test ===")
    pred_single = torch.randn(1, n_samples)
    target_single = torch.randn(1, n_samples)
    loss_single = loss_fn(pred_single, target_single)
    print(f"Single sample loss: {loss_single.item():.6f}")
